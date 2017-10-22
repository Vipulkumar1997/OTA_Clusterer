#!/bin/bash
sysctl -w vm.max_map_count=262144
docker run -d --memory-reservation="4g" -p 8080:8080 -p 9200:9200 --name fess codelibs/fess:latest 
docker run -d -p 5601:5601 -e ELASTICSEARCH_URL=http://172.17.0.2:9200 --name kibana kibana:latest
echo "setup completed"
