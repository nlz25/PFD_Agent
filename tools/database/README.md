## Environment variables
Please set following variables
```env
DATABASE_SERVER_WORK_PATH = "SERVER_WORK_PATH"
INFO_DB_PATH = "PATH_TO_NODE_INFO_DB"
```
Untar the dataset file `database/domain_datasets.tar.gz`, the domain_datasets directory has following trees
```Bash
domain_datasets/
├── datasets/
│   ├── SemiCond.db
│   └── Cluster.db
└── node_info.db
```
The database tool would query the `nodel_info.db` to check for the metadata of each dataset. 
