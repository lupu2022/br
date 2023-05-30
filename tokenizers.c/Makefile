.PHONY: all

all:
	mkdir -p ../install/lib
	mkdir -p ../install/include/tokenizers
	cp include/* ../install/include/tokenizers/
	cd rust;cargo build --release
	cp rust/target/release/libtokenizers_c.a ../install/lib/

clean:
	rm -rf rust/target

