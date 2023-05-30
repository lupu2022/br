.PHONY: all

all:
	mkdir -p ../install/tokenizers/include
	mkdir -p ../install/tokenizers/lib
	cp include/* ../install/tokenizers/include/ 
	cd rust;cargo build --release
	cp rust/target/release/libtokenizers_c.a ../install/tokenizers/lib/

clean:
	rm -rf rust/target

