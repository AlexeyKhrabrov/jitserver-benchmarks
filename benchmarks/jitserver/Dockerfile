FROM ubuntu:18.04

# Install packages required for SSL
RUN apt-get update && apt-get install -y --no-install-recommends \
		libssl1.1 \
	&& rm -rf "/var/lib/apt/lists/"*

# These directories must be mounted as volumes at runtime
RUN mkdir -p "/opt/ibm/java" "/output/vlogs"

EXPOSE 38400

ENTRYPOINT ["/opt/ibm/java/bin/jitserver"]
