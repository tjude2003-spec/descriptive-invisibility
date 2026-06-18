import csv, random
random.seed(42)
rows_by_type = {}
with open('relations_all.csv') as f:
    for row in csv.DictReader(f):
        t = row['relation_type']
        if t not in rows_by_type: rows_by_type[t] = []
        rows_by_type[t].append(row)
with open('relation_validation_sample.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id','relation_type','person_1','person_2','between_text','context','classification','notes'])
    w.writeheader()
    i = 1
    for rtype, rows in rows_by_type.items():
        n = min(200, len(rows))
        for row in random.sample(rows, n):
            w.writerow({'id': i, 'relation_type': row['relation_type'],
                        'person_1': row['person_1'], 'person_2': row['person_2'],
                        'between_text': row['between_text'], 'context': row['context'],
                        'classification': '', 'notes': ''})
            i += 1
    print(f"Saved {i-1} items")
    for t, r in rows_by_type.items():
        print(f"  {t}: {min(200, len(r))} sampled of {len(r)}")
