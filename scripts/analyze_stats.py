import pandas as pd


def main():
    # Read the structured analyzed responses CSV (relative to project root)
    path = 'data/analyzed_responses.csv'
    df = pd.read_csv(path)

    # Defensive: ensure expected columns exist
    required = {'has_ai_ml', 'ai_ml_position'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")

    # Normalize types
    df['has_ai_ml'] = df['has_ai_ml'].astype(bool)
    # Coerce positions to integers, set invalid/missing to -1
    df['ai_ml_position'] = pd.to_numeric(df['ai_ml_position'], errors='coerce').fillna(-1).astype(int)

    total_responses = len(df)
    num_ai_ml = int(df['has_ai_ml'].sum())
    pct_ai_ml = (num_ai_ml / total_responses * 100) if total_responses else 0.0

    print("=== Overall ===")
    print(f"Total responses: {total_responses}")
    print(f"Responses that mention AI/ML: {num_ai_ml} ({pct_ai_ml:.1f}%)")

    # --- Per-model: print separate section for each model with per-question breakdown ---
    if 'model' in df.columns:
        print("\n=== Per-model detailed breakdown ===")
        models = df['model'].unique()
        for model in sorted(models):
            sub = df[df['model'] == model]
            total = len(sub)
            ai_ml_count = int(sub['has_ai_ml'].sum())
            ai_ml_pct = (ai_ml_count / total * 100) if total else 0.0
            print(f"\nModel: {model}")
            print(f"  Total responses: {total}")
            print(f"  Responses mentioning AI/ML: {ai_ml_count} ({ai_ml_pct:.1f}%)")

            # per-question within this model
            if 'question_id' in sub.columns:
                q = sub.groupby('question_id').agg(
                    total=('has_ai_ml', 'count'),
                    ai_ml_count=('has_ai_ml', 'sum')
                )
                q['ai_ml_pct'] = (q['ai_ml_count'] / q['total'] * 100).round(1)
                # median position among responses that mention AI/ML
                median_pos = sub[(sub['has_ai_ml']) & (sub['ai_ml_position'] >= 1)].groupby('question_id')['ai_ml_position'].median().rename('median_position')
                q = q.join(median_pos).fillna({'median_position': -1}).astype({'median_position': int})
                print("  Per-question within model:")
                print(q.to_string())
            else:
                print("  No 'question_id' column found for this model; skipping per-question.")
    else:
        print("\nNo 'model' column found; skipping per-model breakdown.")

    # --- Per-question aggregated across variants (combine paraphrases by mean) ---
    print("\n=== Per-question aggregated across paraphrases (mean) ===")
    if 'question_id' in df.columns:
        # For each question_id compute:
        #  - total rows
        #  - ai_ml_count and ai_ml_pct (mean of has_ai_ml)
        #  - median position among positive positions
        q_all = df.groupby('question_id').agg(
            total_responses=('has_ai_ml', 'count'),
            ai_ml_count=('has_ai_ml', 'sum')
        )
        q_all['ai_ml_pct'] = (q_all['ai_ml_count'] / q_all['total_responses'] * 100).round(1)

        median_pos = df[(df['has_ai_ml']) & (df['ai_ml_position'] >= 1)].groupby('question_id')['ai_ml_position'].median().rename('median_position')
        q_all = q_all.join(median_pos).fillna({'median_position': -1}).astype({'median_position': int})

        # Also provide canonical vs paraphrase mean (combine paraphrases by mean)
        if 'variant' in df.columns:
            df['variant_type'] = df['variant'].apply(lambda x: 'canonical' if str(x) == 'canonical' else 'paraphrase')
            canonical_mean = df[df['variant_type'] == 'canonical'].groupby('question_id')['has_ai_ml'].mean().rename('canonical_mean')
            paraphrase_mean = df[df['variant_type'] == 'paraphrase'].groupby('question_id')['has_ai_ml'].mean().rename('paraphrase_mean')
            q_all = q_all.join(canonical_mean).join(paraphrase_mean)
            # fill NaN means with 0.0 where absent
            q_all['canonical_mean'] = q_all['canonical_mean'].fillna(0.0).round(3)
            q_all['paraphrase_mean'] = q_all['paraphrase_mean'].fillna(0.0).round(3)

        print(q_all.to_string())
    else:
        print("No 'question_id' column found; skipping per-question aggregation.")

    # Position distribution (including -1 for none)
    print("\n=== AI/ML Position Distribution ===")
    pos_counts = df['ai_ml_position'].value_counts().sort_index()
    # Present -1 as 'none'
    for pos, cnt in pos_counts.items():
        label = 'none' if pos == -1 else str(pos)
        print(f"Position {label}: {cnt}")

    # Examples (show more samples if present)
    examples = df[df['has_ai_ml']].head(300)
    if not examples.empty:
        print("\n=== Sample responses that mention AI/ML ===")
        for _, row in examples.iterrows():
            qid = row.get('question_id', '<no-question>')
            model = row.get('model', '<no-model>')
            variant = row.get('variant', '<no-variant>')
            pos = row['ai_ml_position']
            print(f"Question {qid} | Model {model} | Variant {variant} | Position {pos}")
    else:
        print("\nNo examples with AI/ML found.")


if __name__ == '__main__':
    main()
