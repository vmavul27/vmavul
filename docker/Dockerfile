# Toolchain image for VMAVul (Section 3.4).  Build from the repository root:
#   docker build -f docker/Dockerfile -t vmavul .
# CodeQL is downloaded at build time under the GitHub CodeQL Terms and Conditions
# (https://github.com/github/codeql-cli-binaries/blob/main/LICENSE.md); it is not
# redistributed by this repository.
FROM ubuntu:22.04
ARG LLVM_VERSION=17
ARG CODEQL_VERSION=2.25.4
ARG CPP_QUERIES_VERSION=1.6.2
ARG JOERN_VERSION=4.0.547
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl wget gnupg lsb-release software-properties-common unzip git \
      build-essential python3 python3-venv openjdk-17-jdk-headless bubblewrap \
 && rm -rf /var/lib/apt/lists/*

# Clang / compiler-rt 17.0.6 (ASan, UBSan, LSan) and clang-format
RUN wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh && chmod +x /tmp/llvm.sh \
 && /tmp/llvm.sh ${LLVM_VERSION} \
 && apt-get install -y --no-install-recommends clang-format-${LLVM_VERSION} libclang-rt-${LLVM_VERSION}-dev \
 && rm -rf /var/lib/apt/lists/* /tmp/llvm.sh

# CodeQL CLI + codeql/cpp-queries
RUN curl -L --fail -o /tmp/codeql.zip \
      https://github.com/github/codeql-cli-binaries/releases/download/v${CODEQL_VERSION}/codeql-linux64.zip \
 && unzip -q /tmp/codeql.zip -d /opt && rm /tmp/codeql.zip \
 && /opt/codeql/codeql pack download codeql/cpp-queries@${CPP_QUERIES_VERSION} --dir /opt/qlpacks

# Joern
RUN curl -L --fail -o /tmp/joern.zip \
      https://github.com/joernio/joern/releases/download/v${JOERN_VERSION}/joern-cli.zip \
 && unzip -q /tmp/joern.zip -d /opt && rm /tmp/joern.zip

# Python environment (uv + locked dependencies)
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
WORKDIR /vmavul
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra ml
COPY . .
RUN uv sync --frozen --extra ml

ENV VMAVUL_CLANG=clang-${LLVM_VERSION} \
    VMAVUL_CODEQL=/opt/codeql/codeql \
    VMAVUL_CODEQL_CPP_QUERIES=/opt/qlpacks/codeql/cpp-queries/${CPP_QUERIES_VERSION} \
    VMAVUL_JOERN_HOME=/opt/joern-cli \
    PATH=/vmavul/.venv/bin:${PATH}
CMD ["vmavul", "check-tools", "--config", "configs/vmavul.yaml"]
