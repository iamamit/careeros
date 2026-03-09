# CareerOS Week 1 — LLM Integration Interview Q&A

> **For:** Senior Backend Engineer interviews involving AI/LLM integration  
> **Based on:** Real patterns implemented in CareerOS Week 1  
> **Level:** Mid-Senior (3-7 years experience)

---

## SECTION 1: LLM FUNDAMENTALS

---

### Q1. What is an LLM API and how is it different from a regular REST API?

**Answer:**

An LLM API is a REST API under the hood — you POST a JSON payload and get a JSON response. The differences are:

| Regular REST API | LLM API |
|---|---|
| Response in <100ms | Response in 2–10 seconds |
| Deterministic output | Non-deterministic (same input ≠ same output) |
| Structured response always | Free text by default, needs forcing |
| Simple retry on 5xx | Must handle rate limits, token limits, timeouts |
| Stateless | Stateless but context-aware via message history |

**In CareerOS:**
```python
response = await client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.1,
    max_tokens=2000,
)
content = response.choices[0].message.content
```

**Key insight for interviews:** "We treat the LLM as an external service dependency — same patterns as calling any third-party API, but with extra care around latency, non-determinism, and output validation."

---

### Q2. What is a system prompt and why is it important?

**Answer:**

The system prompt is a privileged instruction sent before the conversation that defines:
- Who the LLM should act as
- What context it should have
- What rules it must follow
- What format to respond in

It persists across the entire conversation and has higher authority than user messages.

**In CareerOS** we inject the candidate profile once in the system prompt:
```python
def _system_prompt():
    return f"""You are a senior technical recruiter evaluating job fit.
CANDIDATE: {settings.candidate_name} | {settings.candidate_title} | {settings.candidate_years_exp}+ years
SKILLS: {settings.candidate_skills}
Be honest and precise. Do not inflate scores."""
```

**Why this matters:** Instead of repeating "Amit has 7 years experience, knows Java, Kafka..." in every single API call, we inject it once in the system prompt. Saves tokens, keeps prompts clean, and ensures consistency.

**Interview follow-up:** "What happens if you put everything in the user message instead?"
Answer: "It works, but it's harder to maintain, wastes tokens, and the model may give less weight to instructions buried in user text versus system context."

---

### Q3. What is temperature in LLMs and how do you choose the right value?

**Answer:**

Temperature controls randomness in the model's output:
- `0.0` → fully deterministic, always picks highest probability token
- `0.1` → near-deterministic, very consistent
- `0.7` → balanced creativity
- `1.0+` → highly creative/random, may go off-topic

**Rule of thumb:**

| Use Case | Temperature | Why |
|---|---|---|
| JSON scoring, extraction | 0.0 – 0.1 | Need consistent, parseable output |
| Analysis, evaluation | 0.1 – 0.3 | Need accuracy with slight variation |
| Cover letters, summaries | 0.6 – 0.8 | Need natural, creative writing |
| Brainstorming, ideation | 0.8 – 1.0 | Need diverse ideas |

**In CareerOS:**
```python
# Job scoring — deterministic, consistent results
await chat_completion_json(..., temperature=0.1)

# Cover letter — more natural, human-sounding
await chat_completion(..., temperature=0.7)

# JD extraction — maximum determinism
await chat_completion_json(..., temperature=0.0)
```

**Interview answer:** "Temperature is the single most impactful parameter for reliability. For anything that needs to be parsed or validated downstream, I keep it at 0.1 or lower. For generative content like cover letters, I go higher to avoid repetitive, robotic output."

---

### Q4. How do you force an LLM to always return valid JSON?

**Answer:**

Three layers of defence:

**Layer 1 — response_format parameter:**
```python
response = await client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=messages,
    response_format={"type": "json_object"},  # forces JSON mode
)
```

**Layer 2 — explicit instruction in system prompt:**
```python
json_system = system_prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation, no preamble."
```

**Layer 3 — fallback extraction if parsing fails:**
```python
try:
    return json.loads(content)
except json.JSONDecodeError:
    # Try to extract JSON from within the response
    match = re.search(r"(\{.*\})", content, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError(f"Could not parse JSON: {content[:200]}")
```

**Layer 4 — Pydantic validation after parsing:**
```python
return JobScoreResponse(
    match_score=raw["match_score"],  # Pydantic validates 0-100 range
    ...
)
```

**Interview answer:** "JSON mode alone is not enough — I've seen models return `json\n{...}` or add a sentence before the JSON even in JSON mode. The defence in depth approach handles every failure mode."

---

### Q5. Explain prompt engineering — what makes a good prompt?

**Answer:**

Prompt engineering is the practice of crafting inputs to get reliable, high-quality outputs from LLMs. Key principles:

**1. Be specific about role and context:**
```
Bad:  "Score this job"
Good: "You are a senior technical recruiter. The candidate has 7 years Java experience..."
```

**2. Define output format explicitly:**
```
Bad:  "Tell me if the candidate is a good fit"
Good: "Return JSON with match_score (0-100), verdict (one sentence), matched_skills array..."
```

**3. Give examples (few-shot prompting):**
```python
system = """
Example output:
{
  "match_score": 82,
  "verdict": "Strong match — Java and Kafka align directly.",
  ...
}
"""
```

**4. Add constraints:**
```
"Do not inflate scores. Be honest. If a skill is missing, mark it missing."
```

**5. Control length:**
```
"Maximum 3 bullet points per section. One sentence per bullet."
```

**In CareerOS** the score prompt explicitly defines the JSON schema the model must return — field by field — leaving zero ambiguity.

**Interview answer:** "Prompt engineering is like writing a very precise specification for a very capable but literal contractor. The more ambiguity you leave, the more variance you get in output. For production systems, every prompt should have: role definition, context injection, exact output format, and constraints."

---

## SECTION 2: PRODUCTION PATTERNS

---

### Q6. How do you implement retry logic for LLM API calls?

**Answer:**

LLMs fail in predictable ways — rate limits (429), timeouts, and temporary service outages. We use **Tenacity** for exponential backoff retry:

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

def llm_retry(func):
    return retry(
        stop=stop_after_attempt(3),           # max 3 attempts
        wait=wait_exponential(min=2, max=8),  # 2s → 4s → 8s
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # re-raise after all attempts exhausted
    )(func)

@llm_retry
async def chat_completion_json(...):
    ...
```

**Why exponential backoff?**
- Rate limits have a time window (e.g. 60 req/min)
- Retrying immediately will just hit the limit again
- Exponential wait gives the window time to reset

**What NOT to retry:**
- 400 Bad Request — your prompt is wrong, retrying won't help
- 401 Unauthorized — wrong API key, retrying won't help
- Only retry transient errors: 429 (rate limit), 503 (temporary outage), timeouts

**Interview answer:** "Retry logic is non-negotiable for LLM services. But it's important to retry only transient errors — retrying a bad request just wastes quota. Tenacity gives us clean declarative retry config with exponential backoff, and we log each retry so we can monitor if rate limiting is becoming a systematic problem."

---

### Q7. How do you implement streaming LLM responses in FastAPI?

**Answer:**

Streaming has two parts — the LLM side and the HTTP side.

**LLM side — enable stream=True:**
```python
stream = await client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=messages,
    stream=True,  # key flag
)
async for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        yield delta  # yield each token as it arrives
```

**FastAPI side — StreamingResponse with async generator:**
```python
@router.post("/score/stream")
async def score_stream(req: JobScoreRequest):
    system_prompt, user_prompt = get_stream_prompt(req)

    async def generate():
        async for token in chat_completion_stream(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
        ):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")
```

**Client side — ReadableStream:**
```javascript
const res = await fetch('/score/stream', { method: 'POST', body: ... });
const reader = res.body.getReader();
while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    output += decoder.decode(value);
}
```

**Interview answer:** "Streaming solves two problems — UX (user sees output immediately instead of waiting 8 seconds) and reliability (avoids HTTP gateway timeouts on long responses). The key is using async generators end-to-end: async for on the LLM stream, yield in the generator, StreamingResponse in FastAPI."

---

### Q8. What is the function calling / tool use pattern and when do you use it?

**Answer:**

Function calling is telling the LLM: "here is a schema — respond by filling it in". The LLM acts as an intelligent parser rather than a free-text generator.

**Classic use cases:**
1. **Structured extraction** — parse unstructured JD into typed fields
2. **Routing** — LLM decides which function/tool to call next
3. **Agent actions** — LLM decides to search web, query DB, call API
4. **Form filling** — extract entities from natural language input

**In CareerOS — JD extraction:**
```python
SYSTEM = """Extract structured requirements from a job description.
Return JSON:
{
  "required_skills": ["Java", "Spring Boot"],
  "preferred_skills": ["AWS"],
  "years_experience": 5,
  "work_type": "hybrid",
  "seniority_level": "senior",
  "key_responsibilities": ["Build microservices", ...]
}"""

raw = await chat_completion_json(
    messages=[{"role": "user", "content": f"Extract from:\n\n{jd_text}"}],
    system_prompt=SYSTEM,
    temperature=0.0,  # maximum determinism for extraction
)
```

**Why temperature=0.0 for extraction?**
Extraction has one correct answer. You don't want creativity — you want precision.

**Interview answer:** "Function calling is the bridge between unstructured AI output and typed backend systems. Instead of the LLM writing an essay, you give it a schema and it fills in the blanks. This is also the foundation of AI agents — the agent loop is essentially: LLM picks a function to call, you execute it, feed result back, LLM picks next function."

---

### Q9. How do you handle LLM errors gracefully in production?

**Answer:**

LLM errors fall into three categories:

**1. Transient errors — retry:**
```python
# Rate limit, timeout, temporary outage
retry_if_exception_type((RateLimitError, APITimeoutError))
```

**2. Structural errors — fallback:**
```python
try:
    result = await score_with_llm(req)
except Exception as e:
    logger.error(f"LLM failed after retries: {e}")
    # Fallback to keyword-based scoring
    result = keyword_score_fallback(req)
```

**3. Output validation errors — reject and log:**
```python
try:
    parsed = JobScoreResponse(**raw)
except ValidationError as e:
    logger.error(f"LLM returned invalid schema: {e}")
    raise HTTPException(500, "AI returned unexpected format")
```

**In streaming — handle mid-stream errors:**
```python
async def chat_completion_stream(...):
    try:
        async for chunk in stream:
            yield chunk
    except RateLimitError:
        yield "\n\n[Rate limit reached — try again shortly]"
    except Exception as e:
        yield f"\n\n[Error: {str(e)}]"
```

**Interview answer:** "Defensive LLM integration has three layers: retry for transient failures, fallback logic for persistent failures, and schema validation for bad output. The worst thing you can do is let LLM failures cascade into 500 errors across your whole system. Fail gracefully, log everything, and always have a fallback."

---

### Q10. What is token limit and how do you manage it?

**Answer:**

Every LLM has a context window — the maximum tokens (roughly words/4) it can process in one call. Groq's llama-3.1-70b has 131,072 tokens.

**Tokens ≈ characters / 4, roughly:**
- 1 page of text ≈ 500 tokens
- Full CV ≈ 800 tokens
- Large JD ≈ 400 tokens
- Our prompt overhead ≈ 300 tokens

**Problems if you exceed limits:**
- API returns 400 error
- Model truncates input silently (dangerous)
- Response gets cut off mid-sentence

**How to manage in CareerOS:**
```python
# Truncate JD if too long before sending
def truncate_jd(jd: str, max_chars: int = 4000) -> str:
    if len(jd) > max_chars:
        return jd[:max_chars] + "\n\n[JD truncated for length]"
    return jd

# Set explicit max_tokens on output
response = await client.chat.completions.create(
    max_tokens=2000,  # cap output tokens
    ...
)
```

**For RAG (Week 2) — chunking solves this:**
Large documents are split into chunks, only relevant chunks are sent to the LLM. This is the core reason RAG exists.

**Interview answer:** "Token limits are a real production constraint. We track estimated token usage before calling, truncate inputs that exceed safe limits, and set explicit max_tokens on output. For documents longer than a few pages, we move to RAG — chunk the document, embed it, retrieve only the relevant pieces, and send those to the LLM."

---

## SECTION 3: ARCHITECTURE & DESIGN

---

### Q11. How do you structure a FastAPI project for LLM services?

**Answer:**

The key principle is **separation of concerns** — same as any microservice:

```
app/
├── routers/      ← HTTP layer only (request parsing, status codes, headers)
├── services/     ← Business logic (prompt building, response parsing)
├── models/       ← Pydantic schemas (request/response contracts)
└── config.py     ← Settings from environment
```

**Router — only HTTP concerns:**
```python
@router.post("/score", response_model=JobScoreResponse)
async def score(req: JobScoreRequest, request: Request):
    api_key = request.headers.get("X-Groq-Api-Key")
    try:
        return await score_job(req, api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Service — owns business logic:**
```python
async def score_job(req: JobScoreRequest, api_key=None) -> JobScoreResponse:
    # Builds prompts, calls LLM, parses response
    raw = await chat_completion_json(...)
    return JobScoreResponse(**raw)
```

**LLM Client — owns infrastructure:**
```python
async def chat_completion_json(...) -> dict:
    # Retry logic, JSON forcing, error handling
```

**Interview answer:** "Same layered architecture I use for any service — routers handle HTTP, services handle business logic, infrastructure clients handle external dependencies. The LLM client is just another infrastructure component, like a database client. Keeping it separate means you can swap Groq for OpenAI or Gemini by changing one file."

---

### Q12. How do you make your LLM service provider-agnostic?

**Answer:**

Abstract the LLM client behind an interface so you can swap providers without changing business logic:

```python
# services/llm.py — one place to change provider
def get_client(api_key: str = None) -> AsyncGroq:
    key = api_key or settings.groq_api_key
    return AsyncGroq(api_key=key)

# To switch to OpenAI — only this file changes:
from openai import AsyncOpenAI
def get_client(api_key: str = None):
    return AsyncOpenAI(api_key=api_key or settings.openai_key)
```

**For multi-provider support:**
```python
def get_client(provider: str = "groq", api_key: str = None):
    if provider == "groq":
        return AsyncGroq(api_key=api_key)
    elif provider == "openai":
        return AsyncOpenAI(api_key=api_key)
    elif provider == "gemini":
        return GoogleGenerativeAI(api_key=api_key)
```

**In CareerOS** — Groq for speed/free tier, with Gemini as fallback since you already built with it in EstimateX.

**Interview answer:** "I abstract the LLM client behind a single function. All business logic calls chat_completion() — it never imports Groq or OpenAI directly. When we needed to add Gemini support for the hackathon, it was a config change, not a code change."

---

### Q13. How do you test LLM-powered services? You can't mock randomness.

**Answer:**

The key insight: **test your code, not the LLM**. The LLM is an external dependency — mock it.

**Unit tests — mock the LLM call:**
```python
@patch("app.services.scorer.chat_completion_json", new_callable=AsyncMock)
def test_score_job(mock_llm):
    # Define exactly what the "LLM" returns
    mock_llm.return_value = {
        "match_score": 82,
        "verdict": "Strong match",
        "matched_skills": [{"skill": "Java", "present": True, "note": "7 years"}],
        ...
    }
    response = client.post("/score", json={...})
    assert response.status_code == 200
    assert response.json()["match_score"] == 82
```

**What you're actually testing:**
- Does the router correctly pass the request to the service?
- Does the service correctly parse the LLM response into a Pydantic model?
- Does Pydantic validation reject invalid scores (e.g. 150/100)?
- Does the error handling work when LLM returns bad JSON?

**Integration tests — test with real API key (CI/CD flag):**
```python
@pytest.mark.integration  # only runs with --integration flag
async def test_real_llm_call():
    result = await score_job(sample_request)
    assert 0 <= result.match_score <= 100
    assert result.apply_recommendation in ["STRONG_YES", "YES", "MAYBE", "NO"]
```

**Interview answer:** "You test the integration layer, not the model. Mock the LLM to return controlled responses and test that your parsing, validation, and error handling work correctly. Separate integration tests — marked and gated — test the real API call in CI with a test key. This way unit tests are fast, free, and deterministic."

---

### Q14. How do you handle sensitive data when using LLM APIs?

**Answer:**

When you send data to an external LLM API, that data leaves your infrastructure. Key concerns:

**1. Never send PII to external LLMs without consent:**
```python
# Bad — sends real user data to third-party
await llm.score(resume_text=user.full_resume)

# Better — anonymise first if needed
await llm.score(resume_text=anonymise_pii(user.full_resume))
```

**2. Use on-premise/private models for sensitive data:**
- Ollama (run llama locally)
- Azure OpenAI (data stays in your Azure tenant)
- AWS Bedrock (data stays in your AWS account)

**3. Groq/OpenAI data policies:**
- Groq: does not use API data for training by default
- OpenAI API: does not train on API data (different from ChatGPT)
- Always check provider's data processing agreement

**4. In CareerOS:**
The API key is never stored server-side — passed per-request in a header, lives only in browser memory. This is our security model for the key itself.

**Interview answer:** "For CareerOS we're processing job descriptions and resumes — not financial or medical data. For a fintech or healthcare product I'd use Azure OpenAI or a self-hosted model so data never leaves the company's infrastructure. The key question is always: what's the classification of the data, and does the LLM provider's data policy align with your compliance requirements?"

---

### Q15. What is the difference between stateless and stateful LLM usage?

**Answer:**

**Stateless (what we built in Week 1):**
Each API call is independent. No memory of previous calls.
```python
# Every call is fresh — no history
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Score this job: ..."}
]
```

**Stateful (conversational, Week 3+):**
Pass the full conversation history in every call.
```python
# Build up conversation history
conversation = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Which of my saved jobs fits my Kafka experience?"},
    {"role": "assistant", "content": "Based on your profile, Job X at Company Y..."},
    {"role": "user", "content": "Tell me more about the salary for that one"},
]
# LLM sees full context — knows what "that one" refers to
response = await client.chat.completions.create(messages=conversation)
```

**The important nuance:**
LLMs themselves are stateless — they have no memory. "Statefulness" is an illusion you create by passing history in every request. Your application manages the state.

**Interview answer:** "LLMs are fundamentally stateless — they process whatever you send and forget it. For single-turn tasks like scoring a JD, stateless is perfect and cheaper. For conversational features like 'tell me more about that job', you maintain conversation history in your application layer and replay it on every call. The cost implication is important — longer history = more tokens = higher cost per call."

---

## SECTION 4: ADVANCED TOPICS

---

### Q16. What is RAG and why is it needed? (Preview of Week 2)

**Answer:**

**RAG = Retrieval Augmented Generation**

The problem it solves: LLMs have a knowledge cutoff and don't know about your private data. You can't dump 1000 job descriptions into a prompt — it would exceed token limits and cost a fortune.

**Without RAG:**
```
User: "Which of my 500 saved jobs best matches my Kafka experience?"
Problem: Can't fit 500 job descriptions into one prompt
```

**With RAG:**
```
1. Store all 500 JDs as vector embeddings in pgvector
2. User asks question → convert question to embedding
3. Find top 5 most semantically similar JDs via vector search
4. Send only those 5 JDs to the LLM
5. LLM answers with relevant context
```

**The pipeline:**
```
Document → Chunk → Embed → Store in pgvector
                                    ↓
Query → Embed → Vector Search → Top K chunks → LLM → Answer
```

**Why pgvector over a dedicated vector DB?**
- You already have PostgreSQL
- No new infrastructure to manage
- SQL + vector search in one query
- Good enough for millions of vectors

**Interview answer:** "RAG solves the context window problem for private data. Instead of stuffing everything into the prompt, you pre-compute embeddings for all your documents, store them in a vector database like pgvector, and at query time retrieve only the semantically relevant chunks. It's the difference between giving the LLM a library card versus copying the entire library into the prompt."

---

### Q17. What are embeddings and how do they work?

**Answer:**

An embedding is a numerical vector representation of text that captures semantic meaning. Similar meaning = similar vector = small distance in vector space.

```python
# "Java developer" and "backend engineer" are semantically similar
# Their embeddings will be close in vector space

embedding_1 = embed("Java developer")       # [0.2, 0.8, -0.3, ...]
embedding_2 = embed("backend engineer")     # [0.19, 0.78, -0.31, ...]
embedding_3 = embed("deep sea fishing")     # [-0.9, 0.1, 0.7, ...]

cosine_similarity(embedding_1, embedding_2)  # 0.97 — very similar
cosine_similarity(embedding_1, embedding_3)  # 0.12 — very different
```

**In practice:**
```python
# Generate embedding via API call
response = await openai.embeddings.create(
    model="text-embedding-3-small",
    input="Senior Java engineer with Kafka experience"
)
vector = response.data[0].embedding  # list of 1536 floats
```

**Storing in pgvector:**
```sql
CREATE EXTENSION vector;
CREATE TABLE job_embeddings (
    id SERIAL PRIMARY KEY,
    job_id INT,
    content TEXT,
    embedding vector(1536)
);

-- Semantic search: find 5 most similar jobs
SELECT job_id, content,
       1 - (embedding <=> query_embedding) AS similarity
FROM job_embeddings
ORDER BY embedding <=> query_embedding
LIMIT 5;
```

**Interview answer:** "Embeddings are the bridge between human language and mathematics. They let you do semantic search — finding similar meaning rather than exact keyword matches. This is why 'Java developer' and 'backend engineer' score as similar when matching a resume, even though they share no exact keywords."

---

### Q18. How would you implement caching for LLM responses?

**Answer:**

LLM calls are expensive (cost + latency). Many queries are repeated or similar. Cache aggressively.

**Level 1 — Exact match cache (Redis):**
```python
import hashlib, json
from redis.asyncio import Redis

redis = Redis()

async def cached_llm_call(prompt: str, ...) -> dict:
    # Create deterministic cache key from prompt
    cache_key = "llm:" + hashlib.md5(prompt.encode()).hexdigest()

    # Check cache first
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # Cache miss — call LLM
    result = await chat_completion_json(prompt, ...)

    # Store with TTL (e.g. 24 hours)
    await redis.setex(cache_key, 86400, json.dumps(result))
    return result
```

**Level 2 — Semantic cache (for similar queries):**
```python
# Instead of exact key, find cached responses with similar embeddings
# If someone asked "score this Java job" and then "evaluate this Java role"
# Both get the same cached answer if similarity > 0.95
```

**What to cache vs not cache:**
| Cache | Don't Cache |
|---|---|
| JD extraction results | Cover letters (personalised) |
| Company info lookups | Real-time job scoring (scores change) |
| Skill classification | Streaming responses |

**Cost impact in production:**
- Groq free tier: 14,400 requests/day
- If 30% of requests are cached: effectively 20,000 requests/day
- At $0.05/1K tokens (OpenAI): caching saves real money at scale

**Interview answer:** "Redis exact-match caching gives you the biggest ROI for LLM cost reduction. Hash the full prompt as a cache key, set a reasonable TTL based on how often the underlying data changes. For a job scoring service, a JD doesn't change, so cache it for 24 hours. Cover letters are personalised so never cache them."

---

### Q19. How would you monitor LLM services in production?

**Answer:**

LLM services need standard API metrics plus AI-specific metrics:

**Standard metrics (Prometheus + Grafana):**
```python
from prometheus_client import Counter, Histogram

llm_calls_total = Counter('llm_calls_total', 'Total LLM API calls', ['endpoint', 'status'])
llm_latency = Histogram('llm_latency_seconds', 'LLM call duration', ['model'])
llm_tokens = Counter('llm_tokens_total', 'Total tokens used', ['type'])  # input/output

# In your LLM client:
with llm_latency.labels(model=settings.groq_model).time():
    response = await client.chat.completions.create(...)

llm_tokens.labels(type='input').inc(response.usage.prompt_tokens)
llm_tokens.labels(type='output').inc(response.usage.completion_tokens)
```

**AI-specific metrics (LangSmith):**
- Per-call traces: what prompt went in, what came out
- Token breakdown per call
- Cost per endpoint
- Output quality over time (did scores drift?)

**Key alerts to set up:**
```yaml
# Alert: LLM error rate > 5%
- alert: LLMHighErrorRate
  expr: rate(llm_calls_total{status="error"}[5m]) > 0.05

# Alert: LLM p95 latency > 15 seconds
- alert: LLMHighLatency
  expr: histogram_quantile(0.95, llm_latency_seconds) > 15

# Alert: Token usage > 80% of daily limit
- alert: TokenQuotaWarning
  expr: llm_tokens_total > 0.8 * daily_token_limit
```

**Interview answer:** "LLM observability has two layers — standard infrastructure metrics via Prometheus (latency, error rate, throughput) and AI-specific tracing via LangSmith (what prompt, what output, how many tokens, what did it cost). The unique challenge is output quality monitoring — you need to sample responses and check if the model is still returning valid schemas after a provider update."

---

### Q20. What is an AI Agent and how does it relate to what we built?

**Answer:**

An AI Agent is a system where an LLM can decide what actions to take, execute them, and loop until it reaches a goal. Think of it as the LLM driving the execution flow rather than just responding.

**The ReAct loop (Reason + Act):**
```
LLM receives task
    ↓
LLM reasons: "I need to fetch the job description first"
    ↓
LLM calls tool: fetch_webpage(url)
    ↓
Tool returns: raw JD text
    ↓
LLM reasons: "Now I need to match against the resume"
    ↓
LLM calls tool: search_resume_rag(query)
    ↓
Tool returns: relevant resume chunks
    ↓
LLM reasons: "Now I have enough to give the final answer"
    ↓
LLM returns: structured match report
```

**What we built in Week 1 is the foundation:**
- Function calling = telling LLM what tools exist → **that's tool definition**
- Structured JSON output = LLM decides what data to return → **that's tool response**
- System prompts = giving LLM context and goals → **that's agent instructions**

**Week 3 — we wire these together with LangChain:**
```python
tools = [
    fetch_webpage_tool,      # fetches job URL content
    search_resume_rag_tool,  # searches your embedded resume
    web_search_tool,         # searches company info
]

agent = create_react_agent(llm, tools)
result = agent.invoke("Analyse this job for me: https://...")
# Agent autonomously fetches JD, searches resume, looks up company, returns report
```

**Interview answer:** "An agent is what happens when you give the LLM a loop instead of a single call. Week 1 gives the LLM a pen — it can write structured output. Week 3 gives the LLM hands — it can call tools, get results, and decide what to do next. The function calling pattern we built is literally the foundation: it's how the agent communicates what action it wants to take."

---

### Q21. How do you handle concurrent LLM requests at scale?

**Answer:**

LLM calls are I/O bound (waiting for network), not CPU bound. This means async/await is the right tool — not threads or multiprocessing.

**FastAPI + async = handles concurrent requests efficiently:**
```python
# These 3 requests run concurrently — not sequentially
async def score_job(req):
    # While waiting for Groq API, other requests are processed
    raw = await chat_completion_json(...)  # non-blocking
    return JobScoreResponse(**raw)
```

**For CPU-intensive tasks alongside LLM — use Celery (Week 4):**
```python
# Queue heavy jobs instead of blocking the request
@celery.task
def bulk_score_jobs(job_ids: list):
    for job_id in job_ids:
        result = asyncio.run(score_job(job_id))
        save_to_db(job_id, result)

# API returns immediately
@router.post("/score/bulk")
async def bulk_score(req: BulkRequest):
    task = bulk_score_jobs.delay(req.job_ids)
    return {"task_id": task.id, "status": "queued"}
```

**Rate limit management for concurrent calls:**
```python
import asyncio

# Semaphore limits concurrent LLM calls to avoid rate limits
semaphore = asyncio.Semaphore(5)  # max 5 concurrent LLM calls

async def rate_limited_llm_call(*args, **kwargs):
    async with semaphore:
        return await chat_completion_json(*args, **kwargs)
```

**Interview answer:** "FastAPI's async model handles LLM concurrency naturally since LLM calls are I/O bound — while one request waits for Groq, others are processed. The real concurrency challenge is respecting the provider's rate limits. We use asyncio.Semaphore to cap concurrent LLM calls, and for bulk operations we offload to Celery so the API stays responsive."

---

### Q22. Compare Groq vs OpenAI vs Gemini for a production backend service.

**Answer:**

| | Groq | OpenAI | Gemini |
|---|---|---|---|
| Speed | 🟢 Fastest (500+ tok/s) | 🟡 Medium | 🟡 Medium |
| Cost | 🟢 Free tier + cheap | 🔴 Most expensive | 🟡 Competitive |
| Reliability | 🟡 Good, newer | 🟢 Most reliable | 🟡 Good |
| Context window | 🟢 131K tokens | 🟢 128K tokens | 🟢 1M tokens |
| JSON mode | 🟢 Supported | 🟢 Supported | 🟡 Via prompting |
| Function calling | 🟢 Supported | 🟢 Best in class | 🟢 Supported |
| Streaming | 🟢 Supported | 🟢 Supported | 🟢 Supported |
| Your experience | 🟢 (CareerOS) | 🟡 | 🟢 (EstimateX) |

**When to use what:**
- **Groq** — development, free tier, speed-critical features (streaming chat)
- **OpenAI GPT-4o** — highest quality reasoning, complex agentic tasks
- **Gemini 1.5 Pro** — massive context window (1M tokens), multimodal, you already know it
- **Self-hosted (Ollama)** — sensitive data, no external API calls

**In CareerOS architecture:**
```python
# Primary: Groq (fast, free)
# Fallback: Gemini (you have EstimateX experience)
provider = settings.llm_provider  # "groq" | "gemini"
client = get_client(provider=provider, api_key=api_key)
```

**Interview answer:** "For CareerOS I chose Groq because it's the fastest available inference and has a generous free tier — ideal for a side project and learning. For production at scale I'd use OpenAI for complex reasoning tasks and Gemini for anything needing large context windows or multimodal input. The architecture abstracts the provider so switching is a config change, not a code change."

---

## QUICK REFERENCE — One-liners for interviews

| Topic | One-liner |
|---|---|
| What is an LLM API? | "A REST API where the response is AI-generated text — same HTTP, different latency and reliability profile" |
| Temperature | "Controls randomness — 0.1 for JSON/scoring, 0.7 for creative writing" |
| Streaming | "Async generator on backend, ReadableStream on frontend — solves timeout and UX problems" |
| Retry logic | "Exponential backoff on rate limits and timeouts only — not on bad requests" |
| JSON output | "response_format + explicit instruction + Pydantic validation = three layers of defence" |
| Function calling | "Give the LLM a schema, it fills it in — the foundation of agents" |
| RAG | "Embed documents, store in pgvector, retrieve relevant chunks at query time — solves token limits for private data" |
| Testing LLMs | "Mock the LLM, test your code — unit tests are fast and free, integration tests are gated" |
| Caching | "Hash the prompt, store in Redis with TTL — cuts costs by 30%+ for repeated queries" |
| Agents | "LLM in a loop with tools — ReAct pattern: reason, act, observe, repeat" |

---

*Built as part of CareerOS — AI-Powered Career Intelligence Platform*  
*Week 1: LLM Integration | Week 2: RAG | Week 3: Agents | Week 4: Production*
