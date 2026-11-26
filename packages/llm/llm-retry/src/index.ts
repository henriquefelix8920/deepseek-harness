/**
 * Provider-routed model-request retry policy on the agent loop's request
 * recovery extension point. Each scheduled retry is durable before its cancellable wait.
 *
 * @module @deepseek-ai/dsh-llm-retry
 */

import { randomUUID } from 'node:crypto'
import type { Context, Events } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { z as zod } from 'zod'
import type { Agent, ModelSelection, RequestErrorAction } from '@deepseek-ai/dsh-agent'
import type { LlmFailure, ResolvedRetryPolicy } from '@deepseek-ai/dsh-llm'
import type {} from '@deepseek-ai/dsh-session-projection'
import { brandString } from '@deepseek-ai/dsh-brand'
import type { RetryId } from './brand.ts'
import type { LlmFailoverEventData, LlmRetryEventData } from './types.ts'

export type { LlmFailoverEventData, LlmRetryEventData, LlmRetryStartedEventData } from './types.ts'
export { RetryId } from './brand.ts'

export const name = 'llm-retry'
export const inject = ['agents', 'sessionProjections']

/** One exact provider/model route the agent may use after a transient failure. */
export interface FailoverTarget {
  /** Registered provider route key. */
  readonly provider: string
  /** Exact model id exposed by that provider. */
  readonly model: string
}

/**
 * Optional provider failover overrides. The built-in OpenRouter → Groq → Gemini
 * → Cerebras route order remains active unless a source provider is overridden.
 */
export interface Config {
  /** Per-source-provider replacement targets, in failover order. */
  readonly failoverRoutes?: Readonly<Record<string, readonly FailoverTarget[]>>
}

const failoverTargetSchema = z.object({
  provider: z.string().min(1).required(),
  model: z.string().min(1).required(),
})

/** Runtime schema for {@link Config}. */
export const Config = z.object({
  failoverRoutes: z.dict(z.array(failoverTargetSchema)),
}) as unknown as z<Config>

/** Default routes use provider-native ids with tool/function calling support. */
const DEFAULT_FAILOVER_ROUTES: Readonly<Record<string, readonly FailoverTarget[]>> = {
  openrouter: [
    { provider: 'groq', model: 'openai/gpt-oss-120b' },
    { provider: 'google', model: 'gemini-3.8-flash' },
    { provider: 'cerebras', model: 'gpt-oss-120b' },
  ],
  groq: [
    { provider: 'google', model: 'gemini-3.8-flash' },
    { provider: 'cerebras', model: 'gpt-oss-120b' },
  ],
  google: [
    { provider: 'cerebras', model: 'gpt-oss-120b' },
  ],
}

const FAILOVER_CODES = new Set([
  'EMPTY_RESPONSE',
  'RATE_LIMIT',
  'QUOTA',
  'SERVER',
  'TIMEOUT',
  'TRANSPORT',
])

/** Select failures for which another provider is a safer recovery than the same route. */
function isFailoverFailure(failure: LlmFailure): boolean {
  return FAILOVER_CODES.has(failure.code)
    || failure.status === 429
    || (failure.status !== undefined && failure.status >= 500)
}

/** Validate route overrides without allowing an unknown executor-level retry policy. */
function validateConfig(config: Config): Config {
  for (const key of Object.keys(config)) {
    if (key === 'retryPolicy') {
      throw new Error('llm-retry: retryPolicy belongs under each provider configuration')
    }
    if (key !== 'failoverRoutes') throw new Error(`llm-retry: unknown key "${key}"`)
  }
  const validated = Config(config)
  for (const [source, targets] of Object.entries(validated.failoverRoutes ?? {})) {
    const seen = new Set<string>()
    for (const target of targets) {
      if (target.provider === source) {
        throw new Error(`llm-retry: failover route "${source}" cannot target itself`)
      }
      if (seen.has(target.provider)) {
        throw new Error(`llm-retry: failover route "${source}" repeats target "${target.provider}"`)
      }
      seen.add(target.provider)
    }
  }
  return validated
}

/** Non-serializable hooks used to make timing policy deterministic in tests. */
export interface RetryInternals {
  /** Random sample in the inclusive zero-to-one range used for jitter. */
  random?: () => number
}

type DownstreamOutcome =
  | { readonly type: 'decision'; readonly decision: RequestErrorAction }
  | { readonly type: 'error'; readonly error: unknown }

async function settleDownstream(
  next: () => Promise<RequestErrorAction>,
): Promise<DownstreamOutcome> {
  try {
    return { type: 'decision', decision: await next() }
  } catch (error: unknown) {
    return { type: 'error', error }
  }
}

function localDelay(config: ResolvedRetryPolicy, retry: number, random: () => number): number {
  const exponent = Math.min(retry - 1, 1024)
  const exponential = Math.min(config.initialDelayMs * 2 ** exponent, config.maxDelayMs)
  const jitter = 1 - config.jitterRatio + 2 * config.jitterRatio * random()
  return Math.min(exponential * jitter, config.maxDelayMs)
}

function retryPolicyKey(policy: ResolvedRetryPolicy): string {
  return policy.mode === 'always'
    ? JSON.stringify([policy.mode, policy.initialDelayMs, policy.maxDelayMs, policy.jitterRatio])
    : JSON.stringify([
      policy.mode,
      policy.maxRetries,
      [...policy.retryableCodes].sort(),
      policy.initialDelayMs,
      policy.maxDelayMs,
      policy.jitterRatio,
    ])
}

function retryStateKey(provider: string, policyKey: string): string {
  return JSON.stringify([provider, policyKey])
}

function cancellableDelay(delayMs: number, signal: AbortSignal): Promise<boolean> {
  if (signal.aborted) return Promise.resolve(false)
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve(true)
    }, delayMs)
    function onAbort(): void {
      clearTimeout(timer)
      resolve(false)
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

interface RetryStateEntry {
  retry: number
  retryId: RetryId
}

type LlmRetryState = Record<string, RetryStateEntry>

// The cast bridges the branded retry id, which Zod cannot express directly.
const llmRetryStateSchema: zod.ZodType<LlmRetryState> = zod.record(zod.string(), zod.object({
  retry: zod.number().int().nonnegative(),
  retryId: zod.string(),
})) as unknown as zod.ZodType<LlmRetryState>

/**
 * Install provider failover and provider-routed normal or unbounded request recovery.
 * Recovery admitted during activation awaits projection registration. A
 * registration failure propagates; cancellation or disposal suppresses retries.
 * @param ctx - plugin context that owns the listener and active waits.
 * @param config - optional failover route overrides; provider registrations own retry policy.
 * @param internals - non-serializable deterministic hooks for tests.
 * @returns when recovery listeners and their projection are ready.
 */
export async function apply(ctx: Context, config: Config = {}, internals: RetryInternals = {}): Promise<void> {
  const validatedConfig = validateConfig(config)
  const failoverRoutes = { ...DEFAULT_FAILOVER_ROUTES, ...validatedConfig.failoverRoutes }
  const projectionReady = ctx.sessionProjections.register({
    key: 'llmRetry',
    stateVersion: 1,
    stateSchema: llmRetryStateSchema,
    init: () => ({}),
    apply: (state, event) => {
      if (event.type === 'step/start' || event.type === 'turn/end') return {}
      if (event.type !== 'llm/retry') return state
      const key = retryStateKey(event.data.provider, event.data.policyKey)
      const entry = state[key]
      if (entry?.retry === event.data.retry && entry.retryId === event.data.retryId) return state
      return { ...state, [key]: { retry: event.data.retry, retryId: event.data.retryId } }
    },
  })
  const random = internals.random ?? Math.random
  const lifetime = new AbortController()
  const active = new Set<Promise<RequestErrorAction>>()

  function track(operation: Promise<RequestErrorAction>): Promise<RequestErrorAction> {
    const tracked = operation.finally(() => active.delete(tracked))
    active.add(tracked)
    return tracked
  }

  async function backoff(
    agent: Agent,
    turn: number,
    step: number,
    failure: LlmFailure,
    provider: string,
    policy: ResolvedRetryPolicy,
    policyKey: string,
    retry: number,
    retryId: RetryId,
    delayMs: number,
    signal: AbortSignal,
  ): Promise<RequestErrorAction> {
    const fusedSignal = AbortSignal.any([signal, lifetime.signal])
    const eventData: LlmRetryEventData = policy.mode === 'normal'
      ? {
        retryId,
        turn,
        step,
        provider,
        mode: policy.mode,
        policyKey,
        retry,
        maxRetries: policy.maxRetries,
        delayMs,
        failure,
      }
      : {
        retryId,
        turn,
        step,
        provider,
        mode: policy.mode,
        policyKey,
        retry,
        delayMs,
        failure,
      }
    agent.session.append('llm/retry', eventData)
    if (!await cancellableDelay(delayMs, fusedSignal)) return
    agent.session.append('llm/retry-started', { retryId, turn, step, retry })
    return { kind: 'retry' }
  }

  async function tryFailover(
    agent: Agent,
    turn: number,
    step: number,
    provider: string,
    failure: LlmFailure,
    signal: AbortSignal,
  ): Promise<RequestErrorAction> {
    if (!isFailoverFailure(failure)) return
    const targets = failoverRoutes[provider]
    if (targets === undefined || targets.length === 0) return
    const selection = agent.model.current()
    if (selection === undefined || selection.provider !== provider) return

    // Failover is monotone within one open step, even if a route override cycles.
    const attempted = new Set<string>([provider])
    for (const event of agent.session.snapshotEvents()) {
      if (event.type !== 'llm/failover' || event.data.turn !== turn || event.data.step !== step) continue
      attempted.add(event.data.fromProvider)
      attempted.add(event.data.toProvider)
    }

    for (const target of targets) {
      if (attempted.has(target.provider)) continue
      if (signal.aborted || lifetime.signal.aborted || ctx.fiber.uid === null) return
      let models
      try {
        // listModels returns no entries for routes whose credentials are not configured.
        models = await agent.ctx.llm.listModels(target.provider)
      } catch {
        continue
      }
      if (signal.aborted || lifetime.signal.aborted || ctx.fiber.uid === null) return
      if (!models.some(model => model.id === target.model)) continue

      const latest = agent.model.current()
      if (latest === undefined || latest.provider !== provider || latest.model !== selection.model) return
      const nextSelection: ModelSelection = { provider: target.provider, model: target.model }
      agent.session.append('llm/failover', {
        turn,
        step,
        fromProvider: provider,
        fromModel: selection.model,
        toProvider: target.provider,
        toModel: target.model,
        failure,
      } satisfies LlmFailoverEventData)
      agent.model.select(nextSelection)
      ctx.logger.info(
        `llm-retry: switching agent "${agent.id}" from "${provider}/${selection.model}" to "${target.provider}/${target.model}" after ${failure.code}`,
      )
      return { kind: 'retry' }
    }
  }

  async function recover(
    { agent, turn, step, provider, failure, retryPolicy: policy, signal }: Parameters<Events['agent/request-error']>[0],
    next: () => Promise<RequestErrorAction>,
  ): Promise<RequestErrorAction> {
    const llmRetryProjectionHandle = await projectionReady
    if (signal.aborted || ctx.fiber.uid === null || lifetime.signal.aborted) return
    if (policy?.mode === 'always') {
      const fusedSignal = AbortSignal.any([signal, lifetime.signal])
      // Preserve downstream-first composition for explicitly unbounded recovery.
      const downstream = await settleDownstream(next)
      if (fusedSignal.aborted) return
      if (downstream.type === 'error') {
        ctx.logger.warn(
          `llm-retry: provider "${provider}" always policy ignored a downstream recovery failure: %o`,
          downstream.error,
        )
      }
      if (downstream.type === 'decision' && downstream.decision?.kind === 'retry') {
        return downstream.decision
      }
      const failover = await tryFailover(agent, turn, step, provider, failure, signal)
      if (failover !== undefined) return failover
      if (signal.aborted || ctx.fiber.uid === null || lifetime.signal.aborted) return
    } else {
      // Failover is independent of the source provider's same-route retry budget.
      const failover = await tryFailover(agent, turn, step, provider, failure, signal)
      if (failover !== undefined) return failover
      if (signal.aborted || ctx.fiber.uid === null || lifetime.signal.aborted) return
      if (policy === undefined) return next()
      if (!policy.retryableCodes.includes(failure.code)) return next()
    }

    if (policy === undefined) return next()
    const policyKey = retryPolicyKey(policy)
    const retryState = llmRetryProjectionHandle.stateOf(agent.session)
    const previous = retryState[retryStateKey(provider, policyKey)]
    const previousRetry = previous?.retry ?? 0
    if (policy.mode === 'normal' && previousRetry >= policy.maxRetries) return next()
    const retry = previousRetry + 1
    const retryId = previous?.retryId ?? brandString<RetryId>(randomUUID())
    let delayMs: number
    if (failure.providerRetryAfterMs !== undefined
      && Number.isFinite(failure.providerRetryAfterMs)
      && failure.providerRetryAfterMs > 0) {
      if (failure.providerRetryAfterMs > policy.maxDelayMs) {
        if (policy.mode === 'normal') return next()
        delayMs = localDelay(policy, retry, random)
      } else {
        delayMs = failure.providerRetryAfterMs
      }
    } else {
      delayMs = localDelay(policy, retry, random)
    }

    return backoff(agent, turn, step, failure, provider, policy, policyKey, retry, retryId, delayMs, signal)
  }

  const disposeListener = ctx.on('agent/request-error', (
    payload,
    next: () => Promise<RequestErrorAction>,
  ) => {
    // A waterfall may have captured this callback before its registration was
    // removed. Lifetime cancellation must prevent that stale callback from
    // entering a downstream policy after disposal.
    if (lifetime.signal.aborted) return Promise.resolve<RequestErrorAction>(undefined)
    return track(recover(payload, next))
  })

  ctx.effect(() => async () => {
    disposeListener()
    lifetime.abort(new Error('llm-retry plugin disposed'))
    await Promise.allSettled([...active])
  }, 'llm-retry: abort and drain active recovery')
  await projectionReady
}
