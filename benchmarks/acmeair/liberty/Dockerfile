FROM websphere-liberty:webProfile7

# Copy required files
COPY --chown=1001:0 "mongo.properties" "server.xml" "/config/"
COPY --chown=1001:0 "acmeair-webapp-2.0.0-SNAPSHOT.war" "/config/apps/"

# Configure Liberty server and cleanup SCC, create vlogs directory
ENV ACMEAIR_PROPERTIES="/config/mongo.properties"
RUN "/opt/ibm/wlp/bin/installUtility" install defaultServer \
	&& "configure.sh" && rm -rf "/output/.classCache" \
	&& mkdir -p "/output/.classCache" "/output/vlogs"

# Set JAVA_HOME to directory that will be mapped to JDK on the host
ENV JAVA_HOME="/opt/ibm/java"
ENV IBM_JAVA_OPTIONS="-Xshareclasses:name=acmeair,cacheDir=/output/.classCache"

# Copy entrypoint script
COPY --chown=1001:0 "run.sh" "/"

# Arguments: mongo_addr mongo_port http_port
ENTRYPOINT ["/run.sh"]
