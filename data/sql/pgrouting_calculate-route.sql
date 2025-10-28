SELECT * FROM app.sodd_locations WHERE location_id IN ('08019', '08230');

SELECT 'origen' AS tipo, id FROM (
    SELECT id FROM app.ways_vertices_pgr 
    ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(2.150980391048032, 41.39933854017455), 4326) 
    LIMIT 1
) AS punto_a
UNION ALL
SELECT 'destino' AS tipo, id FROM (
    SELECT id FROM app.ways_vertices_pgr 
    ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(2.3442124319729665, 41.50578121597784), 4326) 
    LIMIT 1
) AS punto_b;

SELECT 
    r.seq,
    r.node,
    r.edge,
    r.cost,
    r.agg_cost,
    w.name,
    w.length_m,
    w.length_m / 1000 AS length_km,
    w.cost_s,
    w.cost_s / 60 AS cost_minutes,
    w.maxspeed_forward,
    w.maxspeed_backward,
    w.the_geom
FROM pgr_dijkstra(
    'SELECT gid as id, source, target, cost, reverse_cost FROM app.ways',
    42976,
	92986,
    directed := true
) r
LEFT JOIN app.ways w ON r.edge = w.gid
ORDER BY r.seq;