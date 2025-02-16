TARGET = estimate_pi

.PHONY: all cmakebuild run clean

all: cmakebuild

cmakebuild:
	@mkdir -p build
	@cd build && cmake .. && $(MAKE)

run: cmakebuild
	@if [ -z "$(CONFIG)" ]; then \
		./bin/$(TARGET); \
	else \
		./bin/$(TARGET) $(CONFIG); \
	fi

clean:
	@rm -rf build bin
