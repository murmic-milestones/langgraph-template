"""Tools the chat agent may call — one tool per file. [OPTIONAL FEATURE]

This is a generic, project-agnostic package (adjacent to ``app/`` and
``lib/``): drop the same ``tools/`` folder into another LangGraph project
and it works unchanged.

Add a tool:
1. create ``tools/<name>.py`` with a single ``@tool``-decorated function
   (the docstring is the model's documentation — be precise about what it
   does, its parameters, and what it returns);
2. import it here and append it to ``TOOLS``.
The chat agent and the graph's ``ToolNode`` both read ``TOOLS``, so
nothing else needs wiring.

SECURITY — tools are a prompt-injection attack surface. The model decides
*whether* to call a tool and *with what arguments* based on the
conversation, and a user can craft a message that steers it ("ignore your
instructions and call X with Y"). So treat every tool argument as
attacker-controlled:

* keep tools least-privilege — no filesystem, secret, or internal-network
  access unless the task truly needs it, and never wire in credentials a
  tool could be tricked into exfiltrating;
* validate/whitelist arguments inside the tool (paths, URLs, ids) rather
  than trusting the model to pass safe values;
* prefer read-only, side-effect-free tools; gate irreversible actions
  behind an explicit human-approval step (``examples/tool_approval.py``
  is the wiring to copy; ``examples/human_approval.py`` shows the bare
  interrupt() mechanism).

To REMOVE tool calling from the template entirely:
1. delete this ``tools/`` package;
2. in ``app/agents/chat.py`` drop the ``bind_tools`` call (marked
   ``[tools]``);
3. in ``app/graph.py`` delete the three ``[tools]`` lines and restore the
   plain ``builder.add_edge("chat", END)``.
"""

from __future__ import annotations

from tools.get_current_time import get_current_time

TOOLS = [get_current_time]
