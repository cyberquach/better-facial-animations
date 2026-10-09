# build.sh supplies the object targets and configuration. -B keeps each build
# fresh, including release-version and dependency-path changes.
vpath %.cc src/animation src/inference src/plugin src/runtime src/speech

build/%.o: %.cc
	$(CXX) -Wall $(CPPFLAGS) -c $< -o $@

build/better_facial_animations.o: CPPFLAGS += -DBFA_VERSION='"$(BFA_VERSION)"'
build/live_core.o: CPPFLAGS += -isystem "$(FFMPEG)/include"
build/ort_net.o: CPPFLAGS += -isystem "$(ORT)/build/native/include"
