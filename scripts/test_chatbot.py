import httpx, time

BASE = 'http://localhost:8000'
DIVIDER = '='*60
tests = [
    ('Uttar Pradesh',     'tell me about uttar pradesh groundwater'),
    ('Bihar',             'which districts in Bihar are safe'),
    ('Punjab vs Haryana', 'compare Punjab and Haryana groundwater'),
    ('Rajasthan',         'groundwater status of Rajasthan'),
    ('Maharashtra',       'tell me about Maharashtra groundwater'),
]

for label, q in tests:
    print()
    print(DIVIDER)
    print('TEST:  ' + label)
    print('QUERY: ' + q)
    t0 = time.time()
    try:
        r = httpx.post(BASE + '/chat', json={'query': q}, timeout=35)
        elapsed = round(time.time() - t0, 1)
        if r.status_code == 200:
            data = r.json()
            cached = data['cached']
            print('STATUS: OK  |  Time: ' + str(elapsed) + 's  |  Cached: ' + str(cached))
            print('ANSWER:')
            print(data['response'])
        else:
            print('ERROR ' + str(r.status_code) + ': ' + r.text[:300])
    except Exception as e:
        print('EXCEPTION: ' + str(e))
    time.sleep(4)

print()
print(DIVIDER)
print('ALL TESTS DONE')
