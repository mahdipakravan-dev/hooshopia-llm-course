# دورهٔ هوشوپیا: ساخت یه LLM چت‌بات از صفر، بدون API

توی این دوره یه مدل زبانی کوچیک رو قدم‌به‌قدم، با کد واقعی و بدون هیچ API ای، خودمون می‌سازیم:
از جمع‌کردن و تمیزکردن داده، تا توکنایزر، ترنسفورمر، پیش‌آموزش، SFT و یه رابط چت که روی سیستم خودت اجرا می‌شه.

زبان مدل فارسیه؛ ولی همهٔ کد مستقل از زبانه و با عوض کردن داده برای هر زبانی کار می‌کنه.

## برنچ‌ها

هر قسمت یوتیوب یه برنچ داره. هر برنچ روی برنچ قبلی ساخته شده؛ یعنی می‌تونی از هر قسمتی شروع کنی.

| قسمت | برنچ | چی اضافه می‌شه |
|---|---|---|
| ۰ | `00-setup` | ساختار پروژه، محیط Python و PyTorch، تشخیص دستگاه (CPU / CUDA / MPS) |
| ۱ | `01-data` | جمع‌آوری کورپوس، نرمال‌سازی فارسی، فیلتر، حذف تکراری‌ها، شارد کردن |
| ۲ | `02-tokenizer` | توکنایزر BPE بایت‌محور از صفر |
| ۳ | `03-bigram` | مدل bigram، تابع loss، حلقهٔ آموزش (baseline) |
| ۴ | `04-transformer` | Self-Attention با ماسک علّی، Multi-Head، RoPE، بلوک GPT |
| ۵ | `05-pretrain` | داده روی دیسک (memmap)، AdamW، زمان‌بندی LR، mixed precision، checkpoint |
| ۶ | `06-sampling` | Temperature، Top-k، Top-p و KV Cache |
| ۷ | `07-sft` | قالب چت، دیتاست گفتگو، ماسک loss روی پیام کاربر |
| ۸ | `08-eval` | loss و perplexity روی دادهٔ تست، گزارش جواب‌ها |
| ۹ | `09-serve` | سرور FastAPI با استریم و رابط وب چت |

`main` همیشه نسخهٔ کامل آخرین قسمته.

```bash
git clone https://github.com/mahdipakravan-dev/hooshopia-llm-course
cd hooshopia-llm-course
git checkout 00-setup      # یا هر قسمتی که می‌خوای
```

## راه‌اندازی

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m llm.env          # نسخه‌ها و دستگاهی که آموزش روش انجام می‌شه
pytest -q
```

یادداشت هر قسمت توی پوشهٔ [`docs/`](docs) هست.

---
هوشوپیا · هوش مصنوعی برای برنامه‌نویسان · [t.me/hooshopia](https://t.me/hooshopia)
