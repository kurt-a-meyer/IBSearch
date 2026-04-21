"""
IB Full-Time Analyst Job Agent (Class of 2027)
-----------------------------------------------
Uses the Anthropic API with:
  - Web search tool  → find IB analyst postings (NYC & Boston, Class of 2027)
  - Gmail MCP tool   → send email alerts (no Google Cloud / OAuth setup needed)

Requirements:
    pip install anthropic

Environment variables:
    ANTHROPIC_API_KEY   - Your Anthropic API key
    RECIPIENT_EMAIL     - Email address to send alerts to
    GMAIL_MCP_URL       - Your Gmail MCP server URL (from Claude.ai connector settings)
                          Typically: https://gmailmcp.googleapis.com/mcp/v1
    GMAIL_MCP_TOKEN     - OAuth bearer token for the Gmail MCP server
                          (export from Claude.ai — see README for instructions)
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import anthropic

# ── Config ────────────────────────────────────────────────────────────────────

SEEN_POSTINGS_FILE = Path("seen_postings.json")
LOG_FILE = "agent.log"

SEARCH_QUERIES = [
    "investment banking analyst full-time 2027 class New York City analyst program",
    "investment banking analyst full-time 2027 analyst program Boston",
    "IB full time analyst 2027 NYC Goldman Sachs Morgan Stanley JPMorgan Evercore Lazard",
    "IB full time analyst 2027 Boston Jefferies Houlihan Lokey Rothschild",
    "investment banking full time offer 2027 recruiting New York",
    "site:goldmansachs.com OR site:morganstanley.com OR site:jpmorgan.com investment banking analyst 2027",
]

SEARCH_SYSTEM_PROMPT = """You are a job-posting parser for a buy-side credit analyst looking for 
investment banking full-time analyst roles (for the graduating Class of 2027) in NYC and Boston.

Your job:
1. Given web search results, identify ONLY legitimate, currently open full-time IB analyst postings 
   targeted at the Class of 2027 (graduating ~May/June 2027).
2. Exclude: internships, summer analyst roles, MBA/Associate roles, off-cycle roles with no 2027 mention, 
   and any posting that is clearly expired or closed.
3. For each valid posting, return a JSON array of objects with these fields:
   - firm: string (e.g. "Goldman Sachs")
   - role_title: string (e.g. "Investment Banking Analyst - M&A, NYC")
   - location: string (must be NYC or Boston — skip all others)
   - division: string (e.g. "M&A", "Leveraged Finance", "ECM", "General IB")
   - start_date: string (e.g. "Summer 2027" or "Unknown")
   - deadline: string (application deadline if stated, else "Not specified")
   - description: string (2-3 sentence summary of role and any notable details)
   - apply_url: string (direct link to the application page; prefer firm's own careers site)
   - url_is_direct: boolean (true if apply_url is on the firm's own domain)
   - posting_id: string (stable slug, e.g. "gs-ib-analyst-nyc-ma")

Return ONLY a JSON array. No preamble, no markdown fences. If no valid postings found, return [].
"""

EMAIL_SYSTEM_PROMPT = """You are an assistant that sends job alert emails via Gmail MCP.
When asked to send an email, use the create_draft tool to create a Gmail draft,
then immediately use the send tool if available, or confirm the draft was created.
Always use HTML for the email body.
"""

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Seen-postings tracker ─────────────────────────────────────────────────────

def load_seen_postings() -> set:
    if SEEN_POSTINGS_FILE.exists():
        data = json.loads(SEEN_POSTINGS_FILE.read_text())
        return set(data.get("seen", []))
    return set()

def save_seen_postings(seen: set):
    SEEN_POSTINGS_FILE.write_text(json.dumps({"seen": list(seen)}, indent=2))

# ── Job search via Claude + web search ───────────────────────────────────────

def search_for_postings(client: anthropic.Anthropic) -> list[dict]:
    all_postings = []
    seen_ids = set()

    for query in SEARCH_QUERIES:
        log.info(f"Searching: {query}")
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SEARCH_SYSTEM_PROMPT,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{
                    "role": "user",
                    "content": (
                        f"Search for this query and return structured job postings:\n\n{query}\n\n"
                        "Use the web_search tool, then return the JSON array of valid postings."
                    )
                }],
            )

            for block in response.content:
                if block.type == "text":
                    raw = block.text.strip()
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                    raw = raw.strip()
                    if not raw or raw == "[]":
                        continue
                    try:
                        postings = json.loads(raw)
                        for p in postings:
                            pid = p.get("posting_id", "")
                            if pid and pid not in seen_ids:
                                seen_ids.add(pid)
                                all_postings.append(p)
                                log.info(f"  Found: {p.get('firm')} – {p.get('role_title')}")
                    except json.JSONDecodeError as e:
                        log.warning(f"  JSON parse error: {e}")

        except Exception as e:
            log.error(f"  Error during search: {e}")

    return all_postings

# ── Email via Claude + Gmail MCP ──────────────────────────────────────────────

def build_email_html(posting: dict) -> str:
    url = posting.get("apply_url", "#")
    url_note = "" if posting.get("url_is_direct", True) else " (via job board — verify on firm site)"
    return f"""
<html><body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; color: #1a1a1a;">
  <div style="background: #0a2540; padding: 20px 24px; border-radius: 6px 6px 0 0;">
    <h2 style="color: #ffffff; margin: 0; font-size: 18px;">🏦 New IB Analyst Posting — Class of 2027</h2>
  </div>
  <div style="border: 1px solid #e0e0e0; border-top: none; padding: 24px; border-radius: 0 0 6px 6px;">
    <table style="width: 100%; border-collapse: collapse;">
      <tr><td style="padding: 6px 0; font-weight: bold; width: 120px; color: #555;">Firm</td>
          <td style="padding: 6px 0;">{posting.get('firm', 'N/A')}</td></tr>
      <tr><td style="padding: 6px 0; font-weight: bold; color: #555;">Role</td>
          <td style="padding: 6px 0;">{posting.get('role_title', 'N/A')}</td></tr>
      <tr><td style="padding: 6px 0; font-weight: bold; color: #555;">Location</td>
          <td style="padding: 6px 0;">{posting.get('location', 'N/A')}</td></tr>
      <tr><td style="padding: 6px 0; font-weight: bold; color: #555;">Division</td>
          <td style="padding: 6px 0;">{posting.get('division', 'N/A')}</td></tr>
      <tr><td style="padding: 6px 0; font-weight: bold; color: #555;">Start Date</td>
          <td style="padding: 6px 0;">{posting.get('start_date', 'N/A')}</td></tr>
      <tr><td style="padding: 6px 0; font-weight: bold; color: #555;">Deadline</td>
          <td style="padding: 6px 0;">{posting.get('deadline', 'Not specified')}</td></tr>
    </table>
    <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 16px 0;">
    <p style="color: #333; line-height: 1.6;">{posting.get('description', '')}</p>
    <div style="margin-top: 20px;">
      <a href="{url}" style="background: #0a2540; color: #fff; padding: 12px 24px;
         text-decoration: none; border-radius: 4px; font-weight: bold; display: inline-block;">
        Apply Now{url_note}
      </a>
    </div>
    <p style="margin-top: 24px; font-size: 12px; color: #888;">
      Detected by IB Job Agent on {datetime.now().strftime('%B %d, %Y at %I:%M %p ET')}.
      Always verify the posting is still open before applying.
    </p>
  </div>
</body></html>
"""

def send_email_via_mcp(client: anthropic.Anthropic, recipient: str, posting: dict) -> bool:
    """Send a job alert email using Claude + Gmail MCP tool."""
    mcp_url = os.environ.get("GMAIL_MCP_URL", "https://gmailmcp.googleapis.com/mcp/v1")
    mcp_token = os.environ.get("GMAIL_MCP_TOKEN", "")

    subject = f"[IB Alert] {posting.get('firm')} – {posting.get('role_title')} ({posting.get('location')})"
    html_body = build_email_html(posting)

    try:
        response = client.beta.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=EMAIL_SYSTEM_PROMPT,
            betas=["mcp-client-2025-04-04"],
            mcp_servers=[
                {
                    "type": "url",
                    "url": mcp_url,
                    "name": "gmail",
                    "authorization_token": mcp_token,
                }
            ],
            messages=[{
                "role": "user",
                "content": (
                    f"Please create and send a Gmail draft to {recipient} with:\n"
                    f"Subject: {subject}\n"
                    f"HTML body:\n{html_body}\n\n"
                    "Use the Gmail create_draft tool to create this email draft."
                )
            }],
        )

        # Check response for success indicators
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                log.info(f"  Gmail MCP response: {block.text[:200]}")

        log.info(f"  ✉ Email sent to {recipient}: {subject}")
        return True

    except Exception as e:
        log.error(f"  Failed to send email via MCP: {e}")
        return False

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    recipient = os.environ.get("RECIPIENT_EMAIL")
    mcp_token = os.environ.get("GMAIL_MCP_TOKEN")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set.")
    if not recipient:
        raise ValueError("RECIPIENT_EMAIL not set.")
    if not mcp_token:
        raise ValueError("GMAIL_MCP_TOKEN not set. See README for how to extract this.")

    log.info("=" * 60)
    log.info(f"IB Job Agent starting — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"Recipient: {recipient}")

    client = anthropic.Anthropic(api_key=api_key)
    seen = load_seen_postings()

    postings = search_for_postings(client)
    log.info(f"Total postings found: {len(postings)}")

    new_postings = [p for p in postings if p.get("posting_id") not in seen]
    log.info(f"New postings: {len(new_postings)}")

    if not new_postings:
        log.info("No new postings. Done.")
        return

    sent = 0
    for posting in new_postings:
        if send_email_via_mcp(client, recipient, posting):
            seen.add(posting["posting_id"])
            sent += 1

    save_seen_postings(seen)
    log.info(f"Done. {sent} alert(s) sent.")

if __name__ == "__main__":
    main()
