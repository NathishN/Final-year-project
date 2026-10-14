import json

log_path = r'C:\Users\nathi\.gemini\antigravity-ide\brain\4ee3a042-026c-42ed-adc2-331d3f9d5b47\.system_generated\logs\transcript_full.jsonl'
out_path = r'c:\fruit_dataset\outputs\phase52_final_validation\user_prompt_phase52.txt'

with open(log_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        try:
            data = json.loads(line)
            content = data.get('content', '')
            if data.get('type') == 'USER_INPUT' and 'PHASE 5.2' in content:
                with open(out_path, 'w', encoding='utf-8') as out:
                    out.write(content)
                print(f"Extracted Phase 5.2 prompt (length: {len(content)} chars)")
                break
        except Exception as e:
            pass
