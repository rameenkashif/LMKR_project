import Anthropic from "@anthropic-ai/sdk";
import { toolDefinitions, runTool } from "./_tools.js";

const SYSTEM_PROMPT = `You are the geoscience assistant embedded in the LMKR Quantitative Seismic Interpretation dashboard.
You answer questions about the Zamzama field's wells, seismic tie quality, and the V11 machine-learning
property predictions (GR, DT, RHOB, VSH, PHIE, SWE, PHIT, AI, VPVS, POIS, LMRHO, MURHO) using the tools
provided - never guess numeric values, always call a tool to look them up. Be concise and geologically
literal (e.g. explain what a property means only if asked). When reporting model reliability, be honest:
a negative blind R2 means the model is not reliable for that target on the held-out well, not just "lower
accuracy" - say so plainly.`;

const MAX_TOOL_ROUNDS = 6;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }
  if (!process.env.ANTHROPIC_API_KEY) {
    res.status(500).json({ error: "ANTHROPIC_API_KEY is not configured on the server." });
    return;
  }

  const { messages } = req.body || {};
  if (!Array.isArray(messages) || messages.length === 0) {
    res.status(400).json({ error: "Request body must include a non-empty 'messages' array." });
    return;
  }

  const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  const conversation = messages.map((m) => ({ role: m.role, content: m.content }));

  try {
    let finalResponse = null;

    for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
      const response = await anthropic.messages.create({
        model: process.env.CLAUDE_MODEL || "claude-sonnet-5",
        max_tokens: 1024,
        system: SYSTEM_PROMPT,
        tools: toolDefinitions,
        messages: conversation,
      });

      conversation.push({ role: "assistant", content: response.content });
      finalResponse = response;

      const toolUseBlocks = response.content.filter((b) => b.type === "tool_use");
      if (toolUseBlocks.length === 0) break;

      const toolResults = [];
      for (const block of toolUseBlocks) {
        let result;
        try {
          result = await runTool(block.name, block.input);
        } catch (err) {
          result = { error: String(err?.message || err) };
        }
        toolResults.push({ type: "tool_result", tool_use_id: block.id, content: JSON.stringify(result) });
      }
      conversation.push({ role: "user", content: toolResults });
    }

    const replyText =
      (finalResponse?.content || [])
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("\n\n") || "I wasn't able to produce a response - please try rephrasing.";

    res.status(200).json({ reply: replyText, messages: conversation });
  } catch (err) {
    console.error("chat handler error", err);
    res.status(500).json({ error: String(err?.message || err) });
  }
}
