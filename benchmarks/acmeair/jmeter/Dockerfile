FROM ibmjava:8-jre

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
		time \
	&& rm -rf "/var/lib/apt/lists/"*

# Install jmeter
ENV JMETER_VERSION="3.3"
RUN mkdir -p "/jmeter" "/output" "/.classCache/" && cd "/jmeter" \
	&& wget "https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-$JMETER_VERSION.tgz" \
	&& tar -xzf "apache-jmeter-$JMETER_VERSION.tgz" && rm -f "apache-jmeter-$JMETER_VERSION.tgz"

# Set jmeter home, update PATH, set SCC location
ENV JMETER_HOME="/jmeter/apache-jmeter-$JMETER_VERSION"
ENV PATH="$JMETER_HOME/bin:$PATH"
ENV IBM_JAVA_OPTIONS="-Xshareclasses:name=jmeter_acmeair,cacheDir=/.classCache"

# Copy the script to be executed and other needed files
COPY "AcmeAir-v3.jmx" *.csv "$JMETER_HOME/"
COPY *.jar "$JMETER_HOME/lib/ext/"
COPY "applyLoad.sh" "generateReport.sh" "$JMETER_HOME/bin/"
COPY "jmeter.properties" "/"

# Set jmeter properties
RUN cat "/jmeter.properties" >> "$JMETER_HOME/bin/jmeter.properties" \
	&& rm -f "/jmeter.properties"

# Options that can be set at runtime
ENV JURL="acmeair-webapp"
ENV JPORT=9090
ENV JUSERBOTTOM=0
ENV JUSER=199
ENV JTHREADS=1
ENV JDURATION=60
ENV JVM_ARGS=""

# Arguments: liberty_addr summariser_interval latency_data report_data
ENTRYPOINT ["applyLoad.sh"]
