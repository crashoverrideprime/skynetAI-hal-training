#!/usr/bin/env python3
"""Test improved JSON parsing."""
import json, os, requests, re, time

url = 'https://openrouter.ai/api/v1/chat/completions'
headers = {
    'Authorization': f'Bearer {os.environ["OPENROUTER_API_KEY"]}',
    'Content-Type': 'application/json',
}

payload = {
    'model': 'google/gemini-2.5-pro',
    'messages': [
        {'role': 'system', 'content': 'You are a data generator. Return ONLY valid JSON. No markdown, no code fences.'},
        {'role': 'user', 'content': 'Generate a JSON object with a "messages" array for a smart home AI training example. The example should have: 1) system message with persona, 2) user message asking to turn on a light, 3) assistant message with tool_call to HassTurnOn, 4) tool message with response, 5) final assistant message. Return ONLY valid JSON.'}
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
print(f'Raw response:\n{content}')
print('---')

# Improved parsing
text = content.strip()

# Remove markdown code fences
text = re.sub(r'^```(?:json)?\s*\n?', '', text)
text = re.sub(r'\n?```\s*$', '', text)
text = text.strip()

print(f'After fence removal:\n{text[:500]}')
print('---')

try:
    parsed = json.loads(text)
    print('Valid JSON!')
    print(json.dumps(parsed, indent=2)[:1500])
except json.JSONDecodeError as e:
    print(f'JSON parse error: {e}')
    # Try to find JSON object
    brace_count = 0
    start = -1
    for i, c in enumerate(text):
        if c == '{':
            if start == -1:
                start = i
            brace_count += 1
        elif c == '}':
            brace_count -= 1
            if brace_count == 0 and start != -1:
                candidate = text[start:i+1]
                try:
                    parsed = json.loads(candidate)
                    print(f'Found valid JSON from char {start} to {i+1}!')
                    print(json.dumps(parsed, indent=2)[:1500])
                    break
                except:
                    pass
