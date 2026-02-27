# arXiv API Notes

## Endpoint

- Base endpoint: `https://export.arxiv.org/api/query`
- Protocol: Atom XML feed
- Common request keys: `search_query`, `start`, `max_results`, `sortBy`, `sortOrder`

## Query Pattern in This Skill

- Category scope: one or more categories, default `cs.AI`.
- Keyword scope: `all:<term>` for single words, `all:"multi word term"` for phrases.
- Full pattern in `mode=any`: `cat:<category> AND (<term1> OR <term2> ...)`
- Full pattern in `mode=all`: `cat:<category> AND (<term1> AND <term2> ...)`

## Returned Fields Used by Script

- `atom:id` -> parse arXiv ID and version
- `atom:title` -> paper title
- `atom:summary` -> abstract text
- `atom:author/atom:name` -> author list
- `atom:published` -> publication date
- `atom:updated` -> updated date
- `atom:category@term` -> category tags

## Practical Notes

- Use a meaningful User-Agent in automated jobs.
- Keep `max_results` moderate (for example 20-100) and paginate with `start`.
- Apply local keyword filtering after API fetch because query grammar is broad.
- For multi-category search, issue one query per category and merge by arXiv ID.
