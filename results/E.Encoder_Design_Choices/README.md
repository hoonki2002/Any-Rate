# Any-Rate: encoder design choices

Layout, candidate search, beam width and mode weight comparisons of the Encoder Design Choices appendix, on Qwen2.5-3B.

    proxy/<setting>/l<layer>_proxy.json   relative proxy loss of each mode for the 7 matrices of each layer (key = planes read)
    ppl/<setting>/                        WikiText-2 PPL of the fully encoded model
    tables.py                             layout, beam width and mode weight tables (tables_main.json)
    candidate_search/                     candidate search on 12 matrices and the full-enumeration encoding (table.py)

Settings: default (UHHHHHH, B = 64, weights 1-1-2-3-4), the layouts UUUU, UHHHHU, HHHHHHU, UUHHHH, the beam widths beam16 and beam256,
and the mode weights w<weights>. main is the proxy of the model of the main results (same settings as default, separate run), which the
tables use for the default row; `python tables.py --default default` uses the default run instead.
