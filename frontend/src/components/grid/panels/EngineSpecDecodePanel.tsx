import { SpecDecodeSection } from '@/components/engines/EnginePanelPrimitives'
import { EnginePanelBody } from './EnginePanelBody'
import { engineIdentity, engineLabel } from './engineLabel'
import { EnginePanelNotice, PanelNotice } from './PanelNotice'
import { useEnginePanel } from './useEnginePanel'
import type { PanelContentProps } from '../panelRegistry'

/**
 * How well speculative decoding is paying off: the token acceptance rate, the
 * mean accepted length, and the cumulative draft and accepted counters.
 *
 * The panel is its own placeable type rather than a corner of the cache panel,
 * because on the engines that run it, acceptance is the number that explains
 * the throughput — and on the engines that do not, it is dead space.
 */
export function EngineSpecDecodePanel({ panel }: PanelContentProps) {
  const resolution = useEnginePanel(panel)
  if (resolution.status !== 'resolved' && resolution.status !== 'aggregate') {
    return <EnginePanelNotice resolution={resolution} />
  }

  const { metric } = resolution
  const draftTokens = metric('spec_decode_draft_tokens_total')
  const acceptanceRate = metric('spec_decode_acceptance_rate')
  const meanAcceptanceLength = metric('spec_decode_mean_acceptance_length')

  // Speculative decoding configured: any of the fields speaks for it. vLLM
  // answers with the lifetime counters; SGLang reports only the acceptance
  // gauges and never the counters — gating on the counter alone would tell a
  // drafter-running SGLang engine it was "not using speculative decoding".
  // (Both engines report nothing at all when no drafter is configured, so the
  // negatives below stay honest.)
  //
  // The engine is named on a multi-engine host for the same reason its data
  // would be: with two engines on a page, "this engine" does not say which.
  if (draftTokens === null && acceptanceRate === null && meanAcceptanceLength === null) {
    // Under the aggregate the absence speaks for every model at once, so
    // naming one would be wrong.
    if (resolution.status === 'aggregate') {
      return <PanelNotice>No model is using speculative decoding.</PanelNotice>
    }
    const subject = engineLabel(resolution) ?? 'This engine'
    return <PanelNotice>{`${subject} is not using speculative decoding.`}</PanelNotice>
  }

  // Configured and idle: vLLM's counter arrives at zero before the first
  // draft, and its absence from SGLang's gauges means the same while nothing
  // has been drafted — in both shapes the drafter exists and the wait is the
  // truth, so the panel says "not yet" rather than showing an all-dashes
  // section that looks like a fault.
  if ((draftTokens === 0 || draftTokens === null) && acceptanceRate === null) {
    if (resolution.status === 'aggregate') {
      return <PanelNotice>No model has drafted a token yet.</PanelNotice>
    }
    const subject = engineLabel(resolution) ?? 'This engine'
    return <PanelNotice>{`${subject} has drafted no tokens yet.`}</PanelNotice>
  }

  return (
    <EnginePanelBody
      identity={engineIdentity(resolution)}
      tiles={
        <SpecDecodeSection
          acceptanceRate={metric('spec_decode_acceptance_rate')}
          acceptanceRateLive={metric('spec_decode_acceptance_rate_live')}
          meanAcceptanceLength={metric('spec_decode_mean_acceptance_length')}
          acceptedTokens={metric('spec_decode_accepted_tokens_total')}
          draftTokens={draftTokens}
        />
      }
    />
  )
}
