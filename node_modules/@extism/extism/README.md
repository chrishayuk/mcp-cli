# Extism JS SDK

This is a universal JavaScript SDK for Extism. It works in all the major JavaScript runtimes:

* Browsers (Firefox, Chrome, WebKit)
* Node
* Deno
* Bun

Instead of using FFI and the libextism shared object, this library uses whatever Wasm runtime is already available with the JavaScript runtime.

## Installation

Install via npm:

```shell
$ npm install @extism/extism
```

> **Note**: Keep in mind we will possibly have breaking changes b/w rc versions until we hit 1.0.

## Compatibility

- **Node.js**: `v18+` (with `--experimental-global-webcrypto`); `v20` with no additional flags
- **Deno**: `v1.36+`
- **Bun**: Tested on `v1.0.7`; Bun partially implements WASI.

Browser tests are run using [playwright](https://playwright.dev)'s defaults. In
browsers, background thread support requires `SharedArrayBuffer` and `Atomic`
support. This is only available in
[`crossOriginIsolated`](https://developer.mozilla.org/en-US/docs/Web/API/crossOriginIsolated)
contexts.

## Reference Docs

Reference docs can be found at [https://extism.github.io/js-sdk/](https://extism.github.io/js-sdk/).

## Getting Started

This guide should walk you through some of the concepts in Extism and this JS library.

First you should import `createPlugin` from Extism:
```js
// CommonJS
const createPlugin = require("@extism/extism")

// ES Modules/Typescript
import createPlugin from '@extism/extism';

// Deno
import createPlugin from 'https://raw.githubusercontent.com/extism/js-sdk/main/src/mod.ts';
```

## Creating A Plug-in

The primary concept in Extism is the [plug-in](https://extism.org/docs/concepts/plug-in). You can think of a plug-in as a code module stored in a `.wasm` file.

Plug-in code can come from a file on disk, object storage or any number of places. Since you may not have one handy let's load a demo plug-in from the web:

```js
const plugin = await createPlugin(
    'https://github.com/extism/plugins/releases/latest/download/count_vowels.wasm',
    { useWasi: true }
);
```

> *Note*: Plug-ins can be loaded in a variety of ways. See the reference docs for [createPlugin](https://extism.github.io/js-sdk/functions/createPlugin.html)
> and read about the [manifest](https://extism.org/docs/concepts/manifest/).

## Calling A Plug-in's Exports

We're using a plug-in, `count_vowels`, which was compiled from Rust.
`count_vowels` plug-in does one thing: it counts vowels in a string. As such,
it exposes one "export" function: `count_vowels`. We can call exports using
`Plugin.call`:

```js
let out = await plugin.call("count_vowels", input);
console.log(out.text())

// => {"count": 3, "total": 3, "vowels": "aeiouAEIOU"}
```

All plug-in exports have a simple interface of optional bytes in, and optional
bytes out. This plug-in happens to take a string and return a JSON encoded
string with a report of results.

### Plug-in State

Plug-ins may be stateful or stateless. Plug-ins can maintain state between calls by
the use of variables. Our `count_vowels` plug-in remembers the total number of
vowels it's ever counted in the `total` key in the result. You can see this by
making subsequent calls to the export:

```js
let out = await plugin.call("count_vowels", "Hello, World!");
console.log(out.text())
// => {"count": 3, "total": 9, "vowels": "aeiouAEIOU"}

out = await plugin.call("count_vowels", "Hello, World!");
console.log(out.json())
// => {"count": 3, "total": 9, "vowels": "aeiouAEIOU"}
```

These variables will persist until you call `await plugin.reset()`. Variables
are not shared between plugin instances.

### Configuration

Plug-ins may optionally take a configuration object. This is a static way to
configure the plug-in. Our count-vowels plugin takes an optional configuration
to change out which characters are considered vowels. Example:

```js
const wasm = {
    url: 'https://github.com/extism/plugins/releases/latest/download/count_vowels.wasm'
}

let plugin = await createPlugin(wasm, {
    useWasi: true,
});

let out = await plugin.call("count_vowels", "Yellow, World!");
console.log(out.text())
// => {"count": 3, "total": 3, "vowels": "aeiouAEIOU"}

plugin = await createPlugin(wasm, {
    useWasi: true,
    config: { "vowels": "aeiouyAEIOUY" }
});

out = await plugin.call("count_vowels", "Yellow, World!");
console.log(out.text())
// => {"count": 4, "total": 4, "vowels": "aeiouAEIOUY"}
```

### Host Functions

Let's extend our count-vowels example a little bit: Instead of storing the
`total` in an ephemeral plug-in var, let's store it in a persistent key-value
store!

Wasm can't use our KV store on it's own. This is where [Host
Functions](https://extism.org/docs/concepts/host-functions) come in.

[Host functions](https://extism.org/docs/concepts/host-functions) allow us to
grant new capabilities to our plug-ins from our application. They are simply
some JS functions you write which can be passed down and invoked from any
language inside the plug-in.

Let's load the manifest like usual but load up this `count_vowels_kvstore`
plug-in:

```js
const wasm = {
    url: "https://github.com/extism/plugins/releases/latest/download/count_vowels_kvstore.wasm"
}
```

> *Note*: The source code for this is [here](https://github.com/extism/plugins/blob/main/count_vowels_kvstore/src/lib.rs) and is written in Rust, but it could be written in any of our PDK languages.

Unlike our previous plug-in, this plug-in expects you to provide host functions that satisfy our its import interface for a KV store.

We want to expose two functions to our plugin, `kv_write(key: string, value: Uint8Array)` which writes a bytes value to a key and `kv_read(key: string): Uint8Array` which reads the bytes at the given `key`.
```js
// pretend this is Redis or something :)
let kvStore = new Map();

const options = {
    useWasi: true,
    functions: {
        env: {
            // NOTE: the first argument is always a CurrentPlugin
            kv_read(cp: CurrentPlugin, offs: bigint) {
                const key = cp.read(offs).text();
                let value = kvStore.get(key) ?? new Uint8Array([0, 0, 0, 0]);
                console.log(`Read ${new DataView(value.buffer).getUint32(0, true)} from key=${key}`);
                return cp.store(value);
            },
            kv_write(cp: CurrentPlugin, kOffs: bigint, vOffs: bigint) {
                const key = cp.read(kOffs).text();

                // Value is a PluginOutput, which subclasses DataView. Along
                // with the `text()` and `json()` methods we've seen, we also
                // get DataView methods, such as `getUint32`.
                const value = cp.read(vOffs);
                console.log(`Writing value=${new value.getUint32(0, true)} from key=${key}`);

                kvStore.set(key, value.bytes());
            }
        }
    }
};
```

> *Note*: In order to write host functions you should get familiar with the
> methods on the `CurrentPlugin` type.

We need to pass these imports to the plug-in to create them. All imports of a
plug-in must be satisfied for it to be initialized:

```js
const plugin = await createPlugin(wasm, options);
```

Now we can invoke the event:

```js
let out = await plugin.call("count_vowels", "Hello World!");
console.log(out.text())
// => Read from key=count-vowels"
// => Writing value=3 from key=count-vowels"
// => {"count": 3, "total": 3, "vowels": "aeiouAEIOU"}

out = await plugin.call("count_vowels", "Hello World!");
console.log(out.text())
// => Read from key=count-vowels"
// => Writing value=6 from key=count-vowels"
// => {"count": 3, "total": 6, "vowels": "aeiouAEIOU"}
```

## Run Examples:

```
npm run build

node --experimental-wasi-unstable-preview1 ./examples/node.js wasm/config.wasm

deno run -A ./examples/deno.ts ./wasm/config.wasm

bun run ./examples/node.js wasm/config.wasm
```
