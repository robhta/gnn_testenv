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

KILL 8;

SELECT COUNT(*) FROM new_table3 ;