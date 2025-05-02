# extism js-pdk types

TypeScript definitions for the Extism [JS Plugin Development Kit](https://github.com/extism/js-pdk).

To install these types, add them to your `tsconfig.json`'s `types`:

```json
{
  "compilerOptions": {
    "lib": [], // this ensures unsupported globals aren't suggested
    "types": ["@extism/js-pdk"], // while this makes the IDE aware of the ones that are
  }
}
```

## Development

The JavaScript prelude is defined in `crates/core/prelude`.

## License

BSD-Clause-3
