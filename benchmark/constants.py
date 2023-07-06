# Description: Constants for the benchmark

EVENT = 'datum.com.bbn.tc.schema.avro.cdm18.Event.uuid'

EVENT_SUBJECT = 'datum.com.bbn.tc.schema.avro.cdm18.Event.subject.com.bbn.tc.schema.avro.cdm18.UUID'
EVENT_OBJECT = 'datum.com.bbn.tc.schema.avro.cdm18.Event.predicateObject.com.bbn.tc.schema.avro.cdm18.UUID'
EVENT_OBJECT2 = 'datum.com.bbn.tc.schema.avro.cdm18.Event.predicateObject2.com.bbn.tc.schema.avro.cdm18.UUID'

EVENT_COLUMNS = [EVENT_SUBJECT, EVENT_OBJECT, EVENT_OBJECT2]

EVENT_TIMESTAMP = 'datum.com.bbn.tc.schema.avro.cdm18.Event.timestampNanos'

EVENT_TYPE = 'datum.com.bbn.tc.schema.avro.cdm18.Event.type'

HOST = 'datum.com.bbn.tc.schema.avro.cdm18.Host.uuid'
SUBJECT = 'datum.com.bbn.tc.schema.avro.cdm18.Subject.uuid'
FILE_OBJECT = 'datum.com.bbn.tc.schema.avro.cdm18.FileObject.uuid'
NETFLOW_OBEJCT = 'datum.com.bbn.tc.schema.avro.cdm18.NetFlowObject.uuid'
SRCSINK_OBJECT = 'datum.com.bbn.tc.schema.avro.cdm18.SrcSinkObject.uuid'
UNNAMED_PIPE_OBJECT = 'datum.com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject.uuid'
PRINCIPAL = 'datum.com.bbn.tc.schema.avro.cdm18.Principal.uuid'

NODE_COLUMNS = [HOST, SUBJECT, FILE_OBJECT, NETFLOW_OBEJCT, SRCSINK_OBJECT, UNNAMED_PIPE_OBJECT, PRINCIPAL]

HOST_SUBTYPE = 'datum.com.bbn.tc.schema.avro.cdm18.Host.hostType'
FILE_OBJECT_SUBTYPE = 'datum.com.bbn.tc.schema.avro.cdm18.FileObject.type'
PRINCIPAL_SUBTYPE =  'datum.com.bbn.tc.schema.avro.cdm18.Principal.type'
SRCSINK_SUBTYPE = 'datum.com.bbn.tc.schema.avro.cdm18.SrcSinkObject.type'
SUBJECT_SUBTYPE = 'datum.com.bbn.tc.schema.avro.cdm18.Subject.type'

SUBTYPE_COULMNS = [HOST_SUBTYPE, FILE_OBJECT_SUBTYPE, PRINCIPAL_SUBTYPE, SRCSINK_SUBTYPE, SUBJECT_SUBTYPE]

NODE_COLUMNS_WITH_SUBTYPE = NODE_COLUMNS + SUBTYPE_COULMNS
       



#additonal Edge Information
SUBJECT_PRINCIPAL = 'datum.com.bbn.tc.schema.avro.cdm18.Subject.principalUUID.com.bbn.tc.schema.avro.cdm18.UUID'
UNNAMED_SINK_PIPE_OBJECT = 'datum.com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject.sinkUUID.com.bbn.tc.schema.avro.cdm18.UUID'
UNNAMED_SOURCE_PIPE_OBJECT = 'datum.com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject.sourceUUID.com.bbn.tc.schema.avro.cdm18.UUID'
EVENT_HOST_ID = 'datum.com.bbn.tc.schema.avro.cdm18.Event.hostId'



GT_FILE_CADETS_ENG3 = "/Users/robinbuchta/Documents/GitHub/trustathsh/projekte/secder/Fachliches/darpa_benchmark_lite/ground_truth/eng3/cadets/20180406_1100_cadets_nginx_backdoor.txt"
GT_FILE_CADETS_ENG3_DATE = "20180406"
GT_FILE_CADETS_ENG3_START_TIME = "11:19"
GT_FILE_CADETS_ENG3_END_TIME = "12:10"



# Blacklist of Words for Ground Truth Extraction
GT_EXTRACTION_BLACKLIST = [
    "Event Log:",
    "Adresses:",
    "Interactions:",
    "Files:",
    "Processes:",
    "Connections:",
    "Graph:",

    "->",
    "connect (exploit 80)",
    "connection on port",
    "connection to",
    "read",
    "connect (stage 1)",
    "connect (oc2)",
    "connect (retrecon tcp)",
    "connect (retrecon scp)",
    "write",
    "chmod",
    "inject (devc)",
    "putfile",
    "inject",
    "delete",

    "F1>",
    "F2>",

    "new process (root)",

    "(failed?)",

    "PORT",
    "PID",
] 