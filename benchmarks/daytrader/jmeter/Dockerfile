FROM ibm-semeru-runtimes:open-8-jre-focal

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
		ca-certificates time wget \
	&& rm -rf "/var/lib/apt/lists/"*

# Install jmeter
ENV JMETER_VERSION="3.3"
RUN mkdir -p "/jmeter" "/output" "/.classCache/" && cd "/jmeter" \
	&& wget "https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-$JMETER_VERSION.tgz" \
	&& tar -xzf "apache-jmeter-$JMETER_VERSION.tgz" && rm -f "apache-jmeter-$JMETER_VERSION.tgz"

# Set jmeter home, update PATH, set SCC location
ENV JMETER_HOME="/jmeter/apache-jmeter-$JMETER_VERSION"
ENV PATH="$JMETER_HOME/bin:$PATH"
ENV IBM_JAVA_OPTIONS="-Xshareclasses:name=jmeter_daytrader,cacheDir=/.classCache"

# Copy the script to be executed and other needed files
COPY "daytrader7.jmx" "$JMETER_HOME/"
COPY *.jar "$JMETER_HOME/lib/ext/"
COPY "applyLoad.sh" "generateReport.sh" "$JMETER_HOME/bin/"
COPY "jmeter.properties" "/"

# Set jmeter properties
RUN cat "/jmeter.properties" >> "$JMETER_HOME/bin/jmeter.properties" \
	&& rm -f "/jmeter.properties"

# Options that can be set at runtime
ENV JHOST="localhost"
ENV JPORT=9080
ENV JBOTUID=0
ENV JTOPUID=14999
ENV JSTOCKS=9999
ENV JTHREADS=1
ENV JDURATION=60
ENV JVM_ARGS=""

# Arguments: liberty_addr summariser_interval latency_data report_data
ENTRYPOINT ["applyLoad.sh"]
