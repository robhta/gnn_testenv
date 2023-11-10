-- @block Bookmarked query
-- @group darpa
-- @name Kill Open Conns

/***********
 * STATUS
 */

SHOW STATUS LIKE 'max_used_connections';

SET GLOBAL max_connections = 10;

show status where variable_name = 'threads_connected';
SHOW PROCESSLIST;

SELECT 
CONCAT('KILL ', id, ';') 
FROM INFORMATION_SCHEMA.PROCESSLIST 
WHERE User = 'root' 
AND db = 'darpa';

KILL 72;

SHOW INDEX FROM new_table4 ;


SELECT src_num_new, COUNT(DISTINCT event_type) FROM new_table4 GROUP BY src_num_new;

SELECT src_num_new, event_type, COUNT(*) FROM new_table4 GROUP BY src_num_new, event_type;
SELECT dst_num_new, event_type, COUNT(*) FROM new_table4 GROUP BY dst_num_new, event_type;

SELECT src_num_new as node, event_type, COUNT(*) FROM new_table4 t1 GROUP BY node, event_type
JOIN new_table4 t2 ON t1.src_num_new = t2.dst_num_new;


--- Long Running... # # with index fast 
SELECT src_num_new AS node, count_src.event_type , count_src.count AS src_count, count_dst.count AS dst_count
FROM
(
    SELECT src_num_new, event_type, COUNT(*) AS count
    FROM new_table4
    GROUP BY src_num_new, event_type
) count_src
JOIN
(
    SELECT dst_num_new, event_type, COUNT(*) AS count
    FROM new_table4
    GROUP BY dst_num_new, event_type
) count_dst
ON count_src.src_num_new = count_dst.dst_num_new AND count_src.event_type = count_dst.event_type;

CREATE INDEX IF NOT EXISTS src_num ON new_table4 (src_num);

CREATE INDEX IF NOT EXISTS dst_num ON new_table4 (dst_num);

CREATE INDEX IF NOT EXISTS event_type ON new_table4 (event_type);

SELECT COUNT(*) FROM new_table4 ; --- 12192401
SELECT COUNT(*) FROM new_table3 ; --- 12192401


SELECT DISTINCT node_type FROM threatrace_nodes; --- just 4 1/2 (null)
SELECT DISTINCT dst_type FROM tmp2;  --- just 4 1/2 (null)
SELECT DISTINCT type_s FROM tmp2; --- 1 process 

SELECT DISTINCT whole_type FROM cadets_nodes_01_record; --- 6 file, principal, process, unix, ipc, dir 


SELECT DISTINCT type_d1 FROM tmp1; --- just 4 1/2 (null) file, unix, sprocess, dir
SELECT DISTINCT type_d2 FROM tmp1;  --- --- file, dir, unix socket 
SELECT DISTINCT type_s FROM tmp1; --- process

SELECT COUNT(*) FROM tmp1; ---- 12915596  just 750361 --- 11442040


query_x = f"SELECT id FROM {table_nodes};"
query_y = f"SELECT node_type FROM {table_nodes};"
query_edge_index_src = f"SELECT src_num_new FROM {table_edges};"
query_edge_index_dst = f"SELECT dst_num_new FROM {table_edges};"

query_node_feature_out = f"SELECT src_num_new, event_type, COUNT(*) FROM new_table4 GROUP BY src_num_new, event_type;"
query_node_feature_in = SELECT dst_num_new, event_type, COUNT(*) FROM new_table4 GROUP BY dst_num_new, event_type;


SELECT COUNT(id) FROM threatrace_nodes; --- 950774
SELECT id FROM threatrace_nodes;
SELECT node_type FROM threatrace_nodes; 
SELECT src_num_new FROM threatrace_edges;
SELECT dst_num_new FROM threatrace_edges;

SELECT DISTINCT EVENT_CLOSE_in FROM threatrace_nodes;


SELECT * FROM threatrace_nodes WHERE node_type_mapped = 4;

SELECT * FROM cadets_r03_tt_nodes_test WHERE id = 1;

SELECT * FROM cadets_nodes_03_record WHERE collected_uuid = "C0E60490-62BE-7B5A-BE62-6F2DBA7BA087";



SELECT src_num_new, event_type, COUNT(*) FROM cadets_r01_tt_edges_test GROUP BY src_num_new, event_type;


SELECT DISTINCT event_type FROM cadets_r01_tt_edges_test;

---- 6, 8, 10, 11, 49, 124,
SELECT * FROM cadets_r03_tt_nodes_test WHERE id = 10; ---7= 7A350F75-9945-425D-8599-15CEBD426F06 is inside (File) 6 is not inside: 9B091956-B243-9D58-83B2-D0F2D89DB317
SELECT * FROM  cadets_r03_tt_nodes_test WHERE id = 9; --- FIle
SELECT * FROM cadets_r03_tt_nodes_test WHERE id = 11; --- 11 file
SELECT * FROM cadets_r03_tt_nodes_test WHERE id = 12; --- 11 file
SELECT * FROM cadets_r03_tt_nodes_test WHERE node = '7A350F75-9945-425D-8599-15CEBD426F06'; 
SELECT * FROM cadets_nodes_03_record WHERE collected_uuid = '7A350F75-9945-425D-8599-15CEBD426F06'; --- 7A350F75-9945-425D-8599-15CEBD426F06
SELECT COUNT(*) FROM cadets_edges_03_record WHERE event_predicateobject2_com_bbn_tc_schema_avro_cdm18_uuid = '7A350F75-9945-425D-8599-15CEBD426F06'; ---40k mal als pred2 obj --- 7A350F75-9945-425D-8599-15CEBD426F06
 --- 6= a process uuid = 0C0CC0C3-3680-11E8-BF66-D9AA8AFF4A69
---9EF37E2E-3E80-11E8-A5CB-3FA3753A265A

---9EF37E2E-3E80-11E8-A5CB-3FA3753A265A
---A4DD7C60-3E80-11E8-A5CB-3FA3753A265A
---A55A32F5-3E80-11E8-A5CB-3FA3753A265A
---878A0030-3E80-11E8-A5CB-3FA3753A265A
---8145E23B-3E80-11E8-A5CB-3FA3753A265A
---A54E7444-3E80-11E8-A5CB-3FA3753A265A
SELECT COUNT(*) FROM cadets_r03_tt_edges_test WHERE dst_uuid = '66BF9E3E-E780-9C5B-80E7-EE1A0B9C8A86';


SELECT * FROM cadets_r01_tt_nodes_test WHERE node = '6D545754-37DD-11E8-BF66-D9AA8AFF4A69'; --- 6 -> Subject process
SELECT * FROM cadets_r01_tt_nodes_test WHERE node = '6D545754-37DD-11E8-BF66-D9AA8AFF4A69'; --- 6 -> Subject process
---2 distinct src uuid
---21 distinct dst uuids 
SELECT * FROM cadets_r01_tt_edges_test WHERE src_uuid = '6D545754-37DD-11E8-BF66-D9AA8AFF4A69'; --- 626edges (src) + edges (dst) 4 ---- -> 4 -> connected 4 subjects src_num = 181005 399430
SELECT DISTINCT dst_uuid  FROM cadets_r01_tt_edges_test WHERE src_uuid = '6D545754-37DD-11E8-BF66-D9AA8AFF4A69'; --> 21 --399430

SELECT * FROM cadets_r01_tt_nodes_test WHERE node = '6D545754-37DD-11E8-BF66-D9AA8AFF4A69'; --- 6

SELECT * FROM cadets_r03_tt_nodes_test WHERE node = '6D545754-37DD-11E8-BF66-D9AA8AFF4A69'; --- 6

SELECT * FROM cadets_r03_tt_nodes_test WHERE node = '6D545754-37DD-11E8-BF66-D9AA8AFF4A69'; --- 6

SELECT * FROM cadets_r03_tt_nodes_train LIMIT 10;

SELECT * FROM cadets_r03_tt_edges_train LIMIT 10;

SELECT num as id, node_type as label from cadets_r03_tt_nodes_train LIMIT 10;


SELECT * FROM cadets_edges_01_record LIMIT 10;

