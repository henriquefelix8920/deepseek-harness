import type { LlmFailure } from '@deepseek-ai/dsh-llm/types'
import type { RetryId } from './brand.ts'

export type { RetryId }

declare module '@deepseek-ai/dsh-session/types' {
  interface SessionEventMap {
    /** Durable, non-surface record of one provider-routed retry scheduled after a failed request attempt. */
    'llm/retry': LlmRetryEventData
    /** Durable transition written after a retry wait succeeds and before the next request attempt starts. */
    'llm/retry-started': LlmRetryStartedEventData
    /** Durable record of one model-request failover to another configured provider route. */
    'llm/failover': LlmFailoverEventData
  }
}

/** Durable payload recorded before one provider-routed model-request retry wait. */
export type LlmRetryEventData =
  | {
    retryId: RetryId
    turn: number
    step: number
    provider: string
    mode: 'normal'
    policyKey: string
    retry: number
    maxRetries: number
    delayMs: number
    failure: LlmFailure
  }

  | {
    retryId: RetryId
    turn: number
    step: number
    provider: string
    mode: 'always'
    policyKey: string
    retry: number
    delayMs: number
    failure: LlmFailure
  }

/** Durable transition recorded after one retry delay completes. */
export interface LlmRetryStartedEventData {
  retryId: RetryId
  turn: number
  step: number
  retry: number
}

/** One provider/model route switch admitted after a retryable model-request failure. */
export interface LlmFailoverEventData {
  /** Open turn containing the failed model request. */
  turn: number
  /** Step whose request was switched to another provider. */
  step: number
  /** Provider route that produced the failure. */
  fromProvider: string
  /** Model id that produced the failure. */
  fromModel: string
  /** Provider route selected for the next attempt. */
  toProvider: string
  /** Model id selected for the next attempt. */
  toModel: string
  /** Structured failure that caused the route switch. */
  failure: LlmFailure
}
