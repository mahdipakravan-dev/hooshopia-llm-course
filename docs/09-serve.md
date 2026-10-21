# قسمت ۹: سرو؛ چت‌بات روی سیستم خودت

آخرین قدم: مدل رو پشت یه API محلی می‌ذاریم و یه صفحهٔ چت براش می‌سازیم. هیچ درخواستی از سیستمت بیرون نمی‌ره.

```
مرورگر ──POST /api/chat──► FastAPI ──► render (قالب چت) ──► stream (KV cache) ──► StreamDecoder
   ▲                                                                                    │
   └──────────────── data: {"text": "..."}  (Server-Sent Events)  ◄─────────────────────┘
```

## API (`llm/server.py`)

| مسیر | کار |
|---|---|
| `GET /` | صفحهٔ چت (`llm/static/index.html`) |
| `GET /api/info` | اندازهٔ مدل و طول context |
| `POST /api/chat` | کل تاریخچهٔ گفتگو رو می‌گیره و جواب رو تکه‌تکه استریم می‌کنه |

```json
{"messages": [{"role": "user", "content": "سلام"}], "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 256}
```

جواب با فرمت Server-Sent Events میاد: هر تکه یه خط `data: {"text": "..."}` و آخرش `data: [DONE]`.
API هیچ حالتی نگه نمی‌داره؛ مرورگر هر بار کل گفتگو رو می‌فرسته (مثل APIهای تجاری).

چون مدل بین چند درخواست هم‌زمان قابل اشتراک نیست، تولید با یه قفل یکی‌یکی انجام می‌شه.
برای چند کاربر هم‌زمان باید batching اضافه کرد؛ برای استفادهٔ شخصی همین کافیه.

## صفحهٔ چت

یه فایل HTML ساده، راست‌چین، بدون هیچ کتابخونه یا CDN خارجی؛ پس کاملاً آفلاین کار می‌کنه.
با `fetch` و `ReadableStream` تکه‌ها رو می‌خونه و هم‌زمان نشون می‌ده.

## اجرا

```bash
python -m llm.server --ckpt checkpoints/sft/ckpt.pt
# بعد: http://localhost:8000
```

یا با curl:

```bash
curl -N http://localhost:8000/api/chat -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"سلام"}]}'
```

## کل مسیر، از اول تا آخر

```bash
python -m llm.data --hf wikimedia/wikipedia --hf-config 20231101.fa --out data/clean
python -m llm.tokenizer train --data data/clean --vocab-size 16384 --out tokenizer.json
python -m llm.tokenize_data --data data/clean --tokenizer tokenizer.json --out data/tokens
python -m llm.pretrain --config configs/small.json
python -m llm.sft --ckpt checkpoints/small/ckpt.pt --data <گفتگوها>.jsonl --out checkpoints/sft
python -m llm.evaluate --ckpt checkpoints/small/ckpt.pt checkpoints/sft/ckpt.pt --tokens data/tokens
python -m llm.server --ckpt checkpoints/sft/ckpt.pt
```
