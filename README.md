# [APAI22-LAB08] Using the NE16 accelerator

## Producing simulation logs

To produce simulation logs run the command:

```
make clean all run runner_args="--trace=ne16"
```

If you want to save it to a file, e.g. `ne16.log`, run the command:

```
make clean all run runner_args="--trace=ne16" > ne16.log
```
