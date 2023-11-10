Select * from cadets_edges_01_record WHERE event_type = 'EVENT_FLOWS_TO' LIMIT 10;

Select distinct event_type from cadets_edges_01_record;

-- 3D0A93E7-36EB-11E8-BF66-D9AA8AFF4A69 subject
-- 1633500B-D34B-8058-8BD3-ABB7D880915E 2
-- 1633500B-D34B-8058-8BD3-ABB7D880915E 1 

select * from cadets_nodes_01_record WHERE collected_uuid = '1633500B-D34B-8058-8BD3-ABB7D880915E' limit 3; -> file 
