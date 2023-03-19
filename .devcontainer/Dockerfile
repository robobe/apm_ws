FROM humble:dev 

# ** [Optional] Uncomment this section to install additional packages. **
# install os toolsros
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
    net-tools \
    python3-colcon-common-extensions \
    tmux \
    tmuxp \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=dialog

# install project dependecies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
    ros-humble-mavros \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=dialog


# Set up auto-source of workspace for ros user
ARG WORKSPACE
RUN source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
RUN echo "if [ -f ${WORKSPACE}/install/setup.bash ]; then source ${WORKSPACE}/install/setup.bash; fi" >> /home/user/.bashrc

