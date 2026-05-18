#!/usr/bin/env python3
"""Test the OpenRouter API for generating tool-call examples."""
import json, os, requests, re, time

url = 'https://openrouter.ai/api/v1/chat/completions'
headers = {
    'Authorization': f'Bearer {os.environ["OPENROUTER_API_KEY"]}',
    'Content-Type': 'application/json',
}

payload = {
    'model': 'google/gemini-2.5-pro',
    'messages': [
        {'role': 'system', 'content': 'You are a data generator. Return ONLY valid JSON.'},
        {'role': 'user', 'content': '''Generate a JSON object with a "messages" array for a smart home AI training example. The example should have:
1. A system message with persona
2. A user message asking to turn on a light
3. An assistant message with a tool_call to HassTurnOn
4. A tool message with response
5. A final assistant message with natural language

Return ONLY valid JSON, no other text.

Example format:
{"messages":[{"role":"system","content":"You are HAL..."},{"role":"user","content":"turn on the living room light"},{"role":"assistant","content":null,"tool_calls":[{"id":"tc_abc123","type":"function","function":{"name":"HassTurnOn","arguments":"{\\"name\\": \\"living room light\\"}"}}]},{"role":"tool","tool_call_id":"tc_abc123","content":"ok"},{"role":"assistant","content":"Living room light turned on, Charm."}]}'''}
    ],
    'temperature': 0.7,
    'max_tokens': 1024,
}

print('Calling API...')
t0 = time.time()
resp = requests.post(url, headers=headers, json=payload, timeout=120)
print(f'Status: {resp.status_code} in {time.time()-t0:.1f}s')
data = resp.json()
content = data['choices'][0]['message']['content']
print(f'Response length: {len(content)}')
print(f'Response:\n{content[:2000]}')
print('---')
# Try to parse
try:
    parsed = json.loads(content)
    print('Valid JSON!')
    print(json.dumps(parsed, indent=2)[:1000])
except json.JSONDecodeError as e:
    print(f'JSON parse error: {e}')
    # Try to find JSON in the response
    match = re.search(r'\{.*"messages".*\}', content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            print('Found JSON via regex!')
            print(json.dumps(parsed, indent=2)[:1000])
        except Exception as e2:
            print(f'Still failed: {e2}')
