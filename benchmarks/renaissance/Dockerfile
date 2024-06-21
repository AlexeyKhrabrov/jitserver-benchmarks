FROM ubuntu:18.04

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
		libssl1.1 \
	&& rm -rf "/var/lib/apt/lists/"*

RUN mkdir -p "/opt/ibm/java" "/output/.classCache" "/output/vlogs" "/scratch"
COPY "JITServerPlugin.jar" "renaissance.jar" "run.sh" "/"

# Set JAVA_HOME to directory that will be mapped to JDK on the host
ENV JAVA_HOME="/opt/ibm/java"
ENV IBM_JAVA_OPTIONS="-Xshareclasses:name=renaissance,cacheDir=/output/.classCache"
ENV JVM_ARGS=""

# Arguments: [options] benchmark
ENTRYPOINT ["/run.sh"]
