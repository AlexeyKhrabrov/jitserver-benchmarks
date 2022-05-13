FROM ubuntu:18.04

# These directories must be mounted as volumes at runtime
RUN mkdir -p "/opt/ibm/java" "/output/vlogs"

EXPOSE 38400

ENTRYPOINT ["/opt/ibm/java/bin/jitserver"]
