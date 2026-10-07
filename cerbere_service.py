import logging
import os

from agentguard import AgentGuard

logger = logging.getLogger("luce.cerbere")

collector_url = os.getenv("AGENTGUARD_COLLECTOR_URL", "https://agentguard-aqal.onrender.com")
api_key = os.getenv("AGENTGUARD_API_KEY")

if not api_key:
    raise RuntimeError("AGENTGUARD_API_KEY is missing")

guard = AgentGuard(
    collector_url=collector_url,
    api_key=api_key,
    max_budget=float(os.getenv("AGENTGUARD_MAX_BUDGET", "10.0")),
    block_on_high=True,
    debug=os.getenv("AGENTGUARD_DEBUG", "false").lower() == "true",
    # Never block a Flask worker for minutes polling for a human decision:
    # raise ApprovalRequiredException immediately. Approval IDs are deterministic,
    # so once approved in the collector, replaying the same call goes through.
    wait_for_approval=False,
)

logger.info("Cerbere initialised (collector=%s)", collector_url)
