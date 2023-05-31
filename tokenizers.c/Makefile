.PHONY: all

all:
	mkdir -p ../install/lib
	mkdir -p ../install/include/tokenizers
	cp include/* ../install/include/tokenizers/
	cd rust;cargo build 
	cp rust/target/debug/libtokenizers_c.a ../install/lib/

clean:
	rm -rf rust/target

