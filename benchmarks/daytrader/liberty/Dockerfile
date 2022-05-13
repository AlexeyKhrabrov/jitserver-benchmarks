FROM websphere-liberty:webProfile7

# Copy required files
COPY --chown=1001:0 "server.xml" "bootstrap.properties" "/config/"
COPY --chown=1001:0 "daytrader-ee7.ear" "/config/apps/"
COPY --chown=1001:0 "db2/" "/opt/ibm/wlp/usr/shared/resources/db2/"

# Install and configure Liberty server and cleanup SCC, create vlogs directory
RUN "/opt/ibm/wlp/bin/installUtility" install defaultServer \
	&& "configure.sh" && rm -rf "/output/.classCache" \
	&& mkdir -p "/output/.classCache" "/output/vlogs"

# Set JAVA_HOME to directory that will be mapped to JDK on the host
ENV JAVA_HOME="/opt/ibm/java"
ENV IBM_JAVA_OPTIONS="-Xshareclasses:name=daytrader,cacheDir=/output/.classCache"

# Copy entrypoint script
COPY --chown=1001:0 "run.sh" "/"

# Arguments: db2_addr db2_port http_port jms_port iiop_port
ENTRYPOINT ["/run.sh"]
