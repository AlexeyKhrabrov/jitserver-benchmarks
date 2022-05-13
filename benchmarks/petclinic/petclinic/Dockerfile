FROM ubuntu:18.04

COPY "spring-petclinic-2.3.0.BUILD-SNAPSHOT.jar" "/application.jar"
COPY "run.sh" "/"

#NOTE: jdk directory must be mounted as volume at runtime
RUN mkdir -p "/opt/ibm/java" "/output/.classCache" "/output/vlogs"

# Set JAVA_HOME to directory that will be mapped to JDK on the host
ENV JAVA_HOME="/opt/ibm/java"
ENV IBM_JAVA_OPTIONS="-Xshareclasses:name=petclinic,cacheDir=/output/.classCache"
ENV SERVER_SERVLET_CONTEXT_PATH="/petclinic"
ENV JVM_ARGS=""

EXPOSE 8080

# Arguments: http_port
ENTRYPOINT ["/run.sh"]
