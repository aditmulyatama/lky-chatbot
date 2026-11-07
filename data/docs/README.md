# Source documents

Put one speech or interview per `.txt` file in this folder, using this format:

```
Title: Speech at the National Day Rally
Date: 1986-08-17
Source: https://link-to-where-you-got-it
---
The full text of the speech goes here...
```

Everything above the `---` line is metadata (shown as sources in the app).
Everything below it is the text that gets searched.

Use publicly available speeches and interviews (for example, the National Archives of Singapore).
Do not add the full text of copyrighted books such as his memoirs, because this repo is public.

After adding or changing files, run `python ingest.py` again.
