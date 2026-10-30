# Judge calibration: labelling sheet

For each item, read the question, the sources the model was given and its
answer, then write one label per item in `labels.csv` (same folder):

- `supported`: every factual claim in the answer (what FastAPI does, how an
  API or parameter behaves, code presented as working) is stated in the
  sources or follows directly from them. Rewording, summarising and generic
  connecting text are fine. Saying the sources do not cover the question is
  supported when they indeed do not.
- `unsupported`: at least one claim is not in the sources or contradicts them,
  including code or parameter names the sources do not show, and claiming the
  sources lack something they contain.
- `unsure`: you cannot decide; these items are left out of the agreement.

Judge support only, not whether the answer is helpful or complete. Use what is
in the sources, not what you know about FastAPI. `note` is optional. The
judge's verdicts are not in this folder; do not look at judge runs before
finishing.


## j-21: How to POST a JSON having a single body parameter in FastAPI?

Question: [so_70772733](https://stackoverflow.com/questions/70772733/how-to-post-a-json-having-a-single-body-parameter-in-fastapi) (CC BY-SA 4.0)

<details>
<summary>Source 1: tutorial/body-multiple-params.md (Body - Multiple Parameters { #body-multiple-parameters } > Recap { #recap })</summary>

You can add multiple body parameters to your *path operation function*, even though a request can only have a single body.

But **FastAPI** will handle it, give you the correct data in your function, and validate and document the correct schema in the *path operation*.

You can also declare singular values to be received as part of the body.

And you can instruct **FastAPI** to embed the body in a key even when there is only a single parameter declared.

</details>

<details>
<summary>Source 2: tutorial/body-multiple-params.md (Body - Multiple Parameters { #body-multiple-parameters } > Embed a single body parameter { #embed-a-single-body-parameter })</summary>

Let's say you only have a single `item` body parameter from a Pydantic model `Item`.

By default, **FastAPI** will then expect its body directly.

But if you want it to expect a JSON with a key `item` and inside of it the model contents, as it does when you declare extra body parameters, you can use the special `Body` parameter `embed`:


```Python
item: Item = Body(embed=True)
```
as in:

{* ../../docs_src/body_multiple_params/tutorial005_an_py310.py hl[17] *}

In this case **FastAPI** will expect a body like:


```JSON hl_lines="2"
{
    "item": {
        "name": "Foo",
        "description": "The pretender",
        "price": 42.0,
        "tax": 3.2
    }
}
```
instead of:


```JSON
{
    "name": "Foo",
    "description": "The pretender",
    "price": 42.0,
    "tax": 3.2
}
```

</details>

<details>
<summary>Source 3: tutorial/body-multiple-params.md (Body - Multiple Parameters { #body-multiple-parameters } > Singular values in body { #singular-values-in-body })</summary>

The same way there is a `Query` and `Path` to define extra data for query and path parameters, **FastAPI** provides an equivalent `Body`.

For example, extending the previous model, you could decide that you want to have another key `importance` in the same body, besides the `item` and `user`.

If you declare it as is, because it is a singular value, **FastAPI** will assume that it is a query parameter.

But you can instruct **FastAPI** to treat it as another body key using `Body`:

{* ../../docs_src/body_multiple_params/tutorial003_an_py310.py hl[23] *}

In this case, **FastAPI** will expect a body like:


```JSON
{
    "item": {
        "name": "Foo",
        "description": "The pretender",
        "price": 42.0,
        "tax": 3.2
    },
    "user": {
        "username": "dave",
        "full_name": "Dave Grohl"
    },
    "importance": 5
}
```
Again, it will convert the data types, validate, document, etc.

</details>

<details>
<summary>Source 4: advanced/strict-content-type.md (Strict Content-Type Checking { #strict-content-type-checking })</summary>

By default, **FastAPI** uses strict `Content-Type` header checking for JSON request bodies, this means that JSON requests **must** include a valid `Content-Type` header (e.g. `application/json`) in order for the body to be parsed as JSON.

</details>

<details>
<summary>Source 5: tutorial/request-files.md (Request Files { #request-files } > What is "Form Data" { #what-is-form-data })</summary>

The way HTML forms (`<form></form>`) sends the data to the server normally uses a "special" encoding for that data, it's different from JSON.

**FastAPI** will make sure to read that data from the right place instead of JSON.

/// note | Technical Details

Data from forms is normally encoded using the "media type" `application/x-www-form-urlencoded` when it doesn't include files.

But when the form includes files, it is encoded as `multipart/form-data`. If you use `File`, **FastAPI** will know it has to get the files from the correct part of the body.

If you want to read more about these encodings and form fields, head to the [<abbr title="Mozilla Developer Network">MDN</abbr> web docs for `POST`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST).

///

/// warning

You can declare multiple `File` and `Form` parameters in a *path operation*, but you can't also declare `Body` fields that you expect to receive as JSON, as the request will have the body encoded using `multipart/form-data` instead of `application/json`.

This is not a limitation of **FastAPI**, it's part of the HTTP protocol.

///

</details>

**Answer:**

To POST a JSON having a single body parameter in FastAPI, you should let FastAPI expect the body directly. By default, FastAPI expects the body directly when you only have a single `Item` body parameter from a Pydantic model `Item`.

However, if you want to expect a JSON with a key `item` and inside of it the model contents, you can use the special `Body` parameter `embed=True`.

For example:

```python
from fastapi import FastAPI, Body
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str
    price: float
    tax: float

app = FastAPI()

@app.post("/items/")
async def create_item(item: Item = Body()):
    return item
```

In this example, FastAPI will expect a JSON body like:

```json
{
    "name": "Foo",
    "description": "The pretender",
    "price": 42.0,
    "tax": 3.2
}
```

It will also validate and document the schema for the `Item` model.  Also note that FastAPI uses strict `Content-Type` header checking for JSON request bodies, this means that JSON requests **must** include a valid `Content-Type` header (e.g. `application/json`) in order for the body to be parsed as JSON.

---

## j-22: Why do UVICORN/Starlette/FastAPI spawn more threads when not using "ASYNC" and don't when using "ASYNC"?

Question: [so_72958085](https://stackoverflow.com/questions/72958085/why-do-uvicorn-starlette-fastapi-spawn-more-threads-when-not-using-async-and-d) (CC BY-SA 4.0)

<details>
<summary>Source 1: async.md (Concurrency and async / await { #concurrency-and-async-await } > In a hurry? { #in-a-hurry })</summary>

<abbr title="too long; didn't read"><strong>TL;DR:</strong></abbr>

If you are using third party libraries that tell you to call them with `await`, like:


```Python
results = await some_library()
```
Then, declare your *path operation functions* with `async def` like:


```Python hl_lines="2"
@app.get('/')
async def read_results():
    results = await some_library()
    return results
```
/// note

You can only use `await` inside of functions created with `async def`.

///

If you are using a third party library that communicates with something (a database, an API, the file system, etc.) and doesn't have support for using `await`, (this is currently the case for most database libraries), then declare your *path operation functions* as normally, with just `def`, like:


```Python hl_lines="2"
@app.get('/')
def results():
    results = some_library()
    return results
```
If your application (somehow) doesn't have to communicate with anything else and wait for it to respond, use `async def`, even if you don't need to use `await` inside.

If you just don't know, use normal `def`.

**Note**: You can mix `def` and `async def` in your *path operation functions* as much as you need and define each one using the best option for you. FastAPI will do the right thing with them.

Anyway, in any of the cases above, FastAPI will still work asynchronously and be extremely fast.

But by following the steps above, it will be able to do some performance optimizations.

</details>

<details>
<summary>Source 2: async.md (Concurrency and async / await { #concurrency-and-async-await } > Conclusion { #conclusion })</summary>

Let's see the same phrase from above:

Modern versions of Python have support for **"asynchronous code"** using something called **"coroutines"**, with **`async` and `await`** syntax.

That should make more sense now. ✨

All that is what powers FastAPI (through Starlette) and what makes it have such an impressive performance.

</details>

<details>
<summary>Source 3: release-notes.md (Release Notes > 0.46.0 (2020-01-08))</summary>

Fix typos and tweak configs. PR [#837](https://github.com/tiangolo/fastapi/pull/837).

Add link to Chinese article in [External Links](https://fastapi.tiangolo.com/external-links/). PR [810](https://github.com/tiangolo/fastapi/pull/810) by [@wxq0309](https://github.com/wxq0309).

Implement `OAuth2AuthorizationCodeBearer` class. PR [#797](https://github.com/tiangolo/fastapi/pull/797) by [@kuwv](https://github.com/kuwv).

Update example upgrade in docs main page. PR [#795](https://github.com/tiangolo/fastapi/pull/795) by [@cdeil](https://github.com/cdeil).

Fix callback handling for sub-routers. PR [#792](https://github.com/tiangolo/fastapi/pull/792) by [@jekirl](https://github.com/jekirl).

Fix typos. PR [#784](https://github.com/tiangolo/fastapi/pull/784) by [@kkinder](https://github.com/kkinder).

Add 4 Japanese articles to [External Links](https://fastapi.tiangolo.com/external-links/). PR [#783](https://github.com/tiangolo/fastapi/pull/783) by [@HymanZHAN](https://github.com/HymanZHAN).

Add support for subtypes of main types in `jsonable_encoder`, e.g. asyncpg's UUIDs. PR [#756](https://github.com/tiangolo/fastapi/pull/756) by [@RmStorm](https://github.com/RmStorm).

Fix usage of Pydantic's `HttpUrl` in docs. PR [#832](https://github.com/tiangolo/fastapi/pull/832) by [@Dustyposa](https://github.com/Dustyposa).

Fix Twitter links in docs. PR [#813](https://github.com/tiangolo/fastapi/pull/813) by [@justindujardin](https://github.com/justindujardin).

Add docs for correctly [using FastAPI with Peewee ORM](https://fastapi.tiangolo.com/advanced/sql-databases-peewee/). Including how to overwrite parts of Peewee to correctly handle async threads. PR [#789](https://github.com/tiangolo/fastapi/pull/789).

</details>

<details>
<summary>Source 4: advanced/events.md (Lifespan Events { #lifespan-events } > Lifespan { #lifespan })</summary>

You can define this *startup* and *shutdown* logic using the `lifespan` parameter of the `FastAPI` app, and a "context manager" (I'll show you what that is in a second).

Let's start with an example and then see it in detail.

We create an async function `lifespan()` with `yield` like this:

{* ../../docs_src/events/tutorial003_py310.py hl[16,19] *}

Here we are simulating the expensive *startup* operation of loading the model by putting the (fake) model function in the dictionary with machine learning models before the `yield`. This code will be executed **before** the application **starts taking requests**, during the *startup*.

And then, right after the `yield`, we unload the model. This code will be executed **after** the application **finishes handling requests**, right before the *shutdown*. This could, for example, release resources like memory or a GPU.

/// tip

The `shutdown` would happen when you are **stopping** the application.

Maybe you need to start a new version, or you just got tired of running it. 🤷

///

The first thing to notice, is that we are defining an async function with `yield`. This is very similar to Dependencies with `yield`.

{* ../../docs_src/events/tutorial003_py310.py hl[14:19] *}

The first part of the function, before the `yield`, will be executed **before** the application starts.

And the part after the `yield` will be executed **after** the application has finished.

If you check, the function is decorated with an `@asynccontextmanager`.

That converts the function into something called an "**async context manager**".

{* ../../docs_src/events/tutorial003_py310.py hl[1,13] *}

A **context manager** in Python is something that you can use in a `with` statement, for example, `open()` can be used as a context manager:


```Python
with open("file.txt") as file:
    file.read()
```
In recent versions of Python, there's also an **async context manager**. You would use it with `async with`:


```Python
async with lifespan(app):
    await do_stuff()
```
When you create a context manager or an async context manager like above, what it does is that, before entering the `with` block, it will execute the code before the `yield`, and after exiting the `with` block, it will execute the code after the `yield`.

In our code example above, we don't use it directly, but we pass it to FastAPI for it to use it.

The `lifespan` parameter of the `FastAPI` app takes an **async context manager**, so we can pass our new `lifespan` async context manager to it.

{* ../../docs_src/events/tutorial003_py310.py hl[22] *}

</details>

<details>
<summary>Source 5: benchmarks.md (Benchmarks { #benchmarks } > Benchmarks and speed { #benchmarks-and-speed })</summary>

When you check the benchmarks, it is common to see several tools of different types compared as equivalent.

Specifically, to see Uvicorn, Starlette and FastAPI compared together (among many other tools).

The simpler the problem solved by the tool, the better performance it will get. And most of the benchmarks don't test the additional features provided by the tool.

The hierarchy is like:

**Uvicorn**: an ASGI server

**Starlette**: (uses Uvicorn) a web microframework

**FastAPI**: (uses Starlette) an API microframework with several additional features for building APIs, with data validation, etc.

**Uvicorn**:

Will have the best performance, as it doesn't have much extra code apart from the server itself.

You wouldn't write an application in Uvicorn directly. That would mean that your code would have to include more or less, at least, all the code provided by Starlette (or **FastAPI**). And if you did that, your final application would have the same overhead as having used a framework and minimizing your app code and bugs.

If you are comparing Uvicorn, compare it against Daphne, Hypercorn, uWSGI, etc. Application servers.

**Starlette**:

Will have the next best performance, after Uvicorn. In fact, Starlette uses Uvicorn to run. So, it probably can only get "slower" than Uvicorn by having to execute more code.

But it provides you the tools to build simple web applications, with routing based on paths, etc.

If you are comparing Starlette, compare it against Sanic, Flask, Django, etc. Web frameworks (or microframeworks).

**FastAPI**:

The same way that Starlette uses Uvicorn and cannot be faster than it, **FastAPI** uses Starlette, so it cannot be faster than it.

FastAPI provides more features on top of Starlette. Features that you almost always need when building APIs, like data validation and serialization. And by using it, you get automatic documentation for free (the automatic documentation doesn't even add overhead to running applications, it is generated on startup).

If you didn't use FastAPI and used Starlette directly (or another tool, like Sanic, Flask, Responder, etc) you would have to implement all the data validation and serialization yourself. So, your final application would still have the same overhead as if it was built using FastAPI. And in many cases, this data validation and serialization is the biggest amount of code written in applications.

So, by using FastAPI you are saving development time, bugs, lines of code, and you would probably get the same performance (or better) you would if you didn't use it (as you would have to implement it all in your code).

If you are comparing FastAPI, compare it against a web application framework (or set of tools) that provides data validation, serialization and documentation, like Flask-apispec, NestJS, Molten, etc. Frameworks with integrated automatic data validation, serialization and documentation.

</details>

**Answer:**

According to the provided context, here’s the explanation of why Uvicorn/Starlette/FastAPI spawn more threads when not using "async" and don't when using "async":

*   **Without "async":** When you're using a traditional, synchronous approach (without `async` and `await`), Uvicorn (and Starlette) typically spawns multiple worker processes to handle concurrent requests. Each worker process runs a separate Python interpreter, effectively creating parallelism. This is because a standard Python process can only handle one task at a time.

*   **With "async":** When you're using `async` and `await`, Starlette (and FastAPI) utilizes a technique called "event loop threading." Instead of spawning multiple processes, it uses a single thread with an event loop. The event loop manages multiple "coroutines" (which are lightweight, concurrent functions) that can run concurrently within the same thread. The `async` and `await` keywords enable this cooperative multitasking, allowing the code to efficiently switch between coroutines without the overhead of context switching between processes.

**In essence:**

*   Without `async`, you get true parallelism by using multiple processes.
*   With `async`, you achieve concurrency within a single thread through the event loop and coroutines.

The context also emphasizes that FastAPI will still work asynchronously and be extremely fast, regardless of whether you use `await` or not, but following the steps mentioned above will help optimize performance.

---

## j-23: How to configure FastAPI logging so that it works both with Uvicorn locally and in production?

Question: [so_77001129](https://stackoverflow.com/questions/77001129/how-to-configure-fastapi-logging-so-that-it-works-both-with-uvicorn-locally-and) (CC BY-SA 4.0)

<details>
<summary>Source 1: deployment/manually.md (Run a Server Manually { #run-a-server-manually } > Install the Server Program { #install-the-server-program })</summary>

When you install FastAPI, it comes with a production server, Uvicorn, and you can start it with the `fastapi run` command.

But you can also install an ASGI server manually.

Make sure you create a [virtual environment](../virtual-environments.md), activate it, and then you can install the server application.

For example, to install Uvicorn:

<div class="termy">

```console
$ pip install "uvicorn[standard]"

---> 100%
```
</div>
A similar process would apply to any other ASGI server program.

/// tip

By adding the `standard`, Uvicorn will install and use some recommended extra dependencies.

That including `uvloop`, the high-performance drop-in replacement for `asyncio`, that provides the big concurrency performance boost.

When you install FastAPI with something like `pip install "fastapi[standard]"` you already get `uvicorn[standard]` as well.

///

</details>

<details>
<summary>Source 2: fastapi-cli.md (FastAPI CLI { #fastapi-cli })</summary>

**FastAPI <abbr title="command line interface">CLI</abbr>** is a command line program that you can use to serve your FastAPI app, manage your FastAPI project, and more.

When you install FastAPI (e.g. with `pip install "fastapi[standard]"`), it comes with a command line program you can run in the terminal.

To run your FastAPI app for development, you can use the `fastapi dev` command:

<div class="termy">

```console
$ <font color="#4E9A06">fastapi</font> dev

  <span style="background-color:#009485"><font color="#D3D7CF"> FastAPI </font></span>  Starting development server 🚀

             Searching for package file structure from directories with
             <font color="#3465A4">__init__.py</font> files
             Importing from <font color="#75507B">/home/user/code/</font><font color="#AD7FA8">awesomeapp</font>

   <span style="background-color:#007166"><font color="#D3D7CF"> module </font></span>  🐍 main.py

     <span style="background-color:#007166"><font color="#D3D7CF"> code </font></span>  Importing the FastAPI app object from the module with the
             following code:

             <u style="text-decoration-style:solid">from </u><u style="text-decoration-style:solid"><b>main</b></u><u style="text-decoration-style:solid"> import </u><u style="text-decoration-style:solid"><b>app</b></u>

      <span style="background-color:#007166"><font color="#D3D7CF"> app </font></span>  Using import string: <font color="#3465A4">main:app</font>

   <span style="background-color:#007166"><font color="#D3D7CF"> server </font></span>  Server started at <font color="#729FCF"><u style="text-decoration-style:solid">http://127.0.0.1:8000</u></font>
   <span style="background-color:#007166"><font color="#D3D7CF"> server </font></span>  Documentation at <font color="#729FCF"><u style="text-decoration-style:solid">http://127.0.0.1:8000/docs</u></font>

      <span style="background-color:#007166"><font color="#D3D7CF"> tip </font></span>  Running in development mode, for production use:
             <b>fastapi run</b>

             Logs:

     <span style="background-color:#007166"><font color="#D3D7CF"> INFO </font></span>  Will watch for changes in these directories:
             <b>[</b><font color="#4E9A06">&apos;/home/user/code/awesomeapp&apos;</font><b>]</b>
     <span style="background-color:#007166"><font color="#D3D7CF"> INFO </font></span>  Uvicorn running on <font color="#729FCF"><u style="text-decoration-style:solid">http://127.0.0.1:8000</u></font> <b>(</b>Press CTRL+C to
             quit<b>)</b>
     <span style="background-color:#007166"><font color="#D3D7CF"> INFO </font></span>  Started reloader process <b>[</b><font color="#34E2E2"><b>383138</b></font><b>]</b> using WatchFiles
     <span style="background-color:#007166"><font color="#D3D7CF"> INFO </font></span>  Started server process <b>[</b><font color="#34E2E2"><b>383153</b></font><b>]</b>
     <span style="background-color:#007166"><font color="#D3D7CF"> INFO </font></span>  Waiting for application startup.
     <span style="background-color:#007166"><font color="#D3D7CF"> INFO </font></span>  Application startup complete.
```
</div>
/// tip

For production you would use `fastapi run` instead of `fastapi dev`. 🚀

///

Internally, **FastAPI CLI** uses [Uvicorn](https://www.uvicorn.dev), a high-performance, production-ready, ASGI server. 😎

The `fastapi` CLI will try to detect automatically the FastAPI app to run, assuming it's an object called `app` in a file `main.py` (or a couple other variants).

But you can configure explicitly the app to use.

</details>

<details>
<summary>Source 3: fastapi-cli.md (FastAPI CLI { #fastapi-cli } > `fastapi run` { #fastapi-run })</summary>

Executing `fastapi run` starts FastAPI in production mode.

By default, **auto-reload** is disabled. It also listens on the IP address `0.0.0.0`, which means all the available IP addresses, this way it will be publicly accessible to anyone that can communicate with the machine. This is how you would normally run it in production, for example, in a container.

In most cases you would (and should) have a "termination proxy" handling HTTPS for you on top, this will depend on how you deploy your application, your provider might do this for you, or you might need to set it up yourself.

/// tip

You can learn more about it in the [deployment documentation](deployment/index.md).

///

</details>

<details>
<summary>Source 4: deployment/manually.md (Run a Server Manually { #run-a-server-manually } > Run the Server Program { #run-the-server-program })</summary>

If you installed an ASGI server manually, you would normally need to pass an import string in a special format for it to import your FastAPI application:

<div class="termy">

```console
$ uvicorn main:app --host 0.0.0.0 --port 80

<span style="color: green;">INFO</span>:     Uvicorn running on http://0.0.0.0:80 (Press CTRL+C to quit)
```
</div>
/// note

The command `uvicorn main:app` refers to:

`main`: the file `main.py` (the Python "module").

`app`: the object created inside of `main.py` with the line `app = FastAPI()`.

It is equivalent to:


```Python
from main import app
```
///

Each alternative ASGI server program would have a similar command, you can read more in their respective documentation.

/// warning

Uvicorn and other servers support a `--reload` option that is useful during development.

The `--reload` option consumes much more resources, is more unstable, etc.

It helps a lot during **development**, but you **shouldn't** use it in **production**.

///

</details>

<details>
<summary>Source 5: advanced/behind-a-proxy.md (Behind a Proxy { #behind-a-proxy } > Additional servers { #additional-servers })</summary>

/// warning

This is a more advanced use case. Feel free to skip it.

///

By default, **FastAPI** will create a `server` in the OpenAPI schema with the URL for the `root_path`.

But you can also provide other alternative `servers`, for example if you want *the same* docs UI to interact with both a staging and a production environment.

If you pass a custom list of `servers` and there's a `root_path` (because your API lives behind a proxy), **FastAPI** will insert a "server" with this `root_path` at the beginning of the list.

For example:

{* ../../docs_src/behind_a_proxy/tutorial003_py310.py hl[4:7] *}

Will generate an OpenAPI schema like:


```JSON hl_lines="5-7"
{
    "openapi": "3.1.0",
    // More stuff here
    "servers": [
        {
            "url": "/api/v1"
        },
        {
            "url": "https://stag.example.com",
            "description": "Staging environment"
        },
        {
            "url": "https://prod.example.com",
            "description": "Production environment"
        }
    ],
    "paths": {
            // More stuff here
    }
}
```
/// tip

Notice the auto-generated server with a `url` value of `/api/v1`, taken from the `root_path`.

///

In the docs UI at [http://127.0.0.1:9999/api/v1/docs](http://127.0.0.1:9999/api/v1/docs) it would look like:

<img src="/img/tutorial/behind-a-proxy/image03.png">
/// tip

The docs UI will interact with the server that you select.

///

/// note | Technical Details

The `servers` property in the OpenAPI specification is optional.

If you don't specify the `servers` parameter and `root_path` is equal to `/`, the `servers` property in the generated OpenAPI schema will be omitted entirely by default, which is the equivalent of a single server with a `url` value of `/`.

///

If you don't want **FastAPI** to include an automatic server using the `root_path`, you can use the parameter `root_path_in_servers=False`:

{* ../../docs_src/behind_a_proxy/tutorial004_py310.py hl[9] *}

and then it won't include it in the OpenAPI schema.

</details>

**Answer:**

FastAPI uses Uvicorn for local development and production environments. To configure FastAPI logging effectively across both, you can use the `uvicorn` server and its options.

Here’s a breakdown of how to configure logging with Uvicorn, covering both development and production scenarios:

**1. Development Mode (using `fastapi dev`)**:

*   When running with `fastapi dev`, FastAPI automatically sets up a reloader that watches for file changes and restarts the server.
*   It uses the `uvloop` library for improved performance during development.
*   The logs from `fastapi dev` are usually quite verbose, including detailed information about file changes, imports, and the FastAPI app startup.  This is helpful for debugging.

**2. Production Mode (using `fastapi run`)**:

*   When using `fastapi run`, Uvicorn is running in production mode.
*   **Auto-reload is disabled** by default in production to prevent issues.
*   Uvicorn, by default, logs to standard output (stdout). You can control the level of detail in the logs using the `--log-level` option. Common values are:
    *   `debug`:  The most verbose level, displaying detailed information.
    *   `info`:  Displays informational messages, warnings, and errors.
    *   `warning`: Displays warning messages.
    *   `error`: Displays error messages.
    *   `critical`: Displays critical error messages.

    Example: `fastapi run --log-level info`

*   If you're running in a container (e.g., Docker), you'll likely want to redirect stdout to a file for logging purposes. This is typical for production deployments.

**3. Using the OpenAPI Schema for Servers (Advanced):**

*   FastAPI automatically generates an OpenAPI schema that includes a `servers` section. This allows you to define different URLs for your API in different environments (e.g., staging, production).
*   The `root_path` parameter in the OpenAPI schema is also automatically generated.
*   While you can't directly configure the logging level through the OpenAPI schema, you can control the server URLs defined there, which can be useful for routing requests to the appropriate environment.

**4. Manual Uvicorn Installation (if not using `fastapi run`)**:

*   If you install Uvicorn manually (e.g., `pip install "uvicorn[standard]"`), you'll need to provide the same logging configuration options when running Uvicorn. The command would look something like: `uvicorn main:app --host 0.0.0.0 --port 80 --log-level info`.

**Summary of Options:**

*   `--log-level`:  Controls the verbosity of the logs (debug, info, warning, error, critical).
*   `--reload`:  (Development only) Automatically restarts the server when file changes are detected.  Disable in production.

**Key Considerations:**

*   **Production Logging:** For production deployments, it's *crucial* to log to a file rather than stdout. This allows you to easily monitor your application's behavior and troubleshoot issues without disrupting the service.
*   **Log Rotation:** When logging to files, implement log rotation to prevent the log files from growing indefinitely.

To recap, you manage logging primarily by using the command-line options provided by Uvicorn (such as `--log-level`). In production, you'll redirect the logs to a file and configure the log level accordingly.

---

## j-24: How to get Files and Form data using the Request object in FastAPI?

Question: [so_76970173](https://stackoverflow.com/questions/76970173/how-to-get-files-and-form-data-using-the-request-object-in-fastapi) (CC BY-SA 4.0)

<details>
<summary>Source 1: tutorial/request-forms-and-files.md (Request Forms and Files { #request-forms-and-files } > Define `File` and `Form` parameters { #define-file-and-form-parameters })</summary>

Create file and form parameters the same way you would for `Body` or `Query`:

{* ../../docs_src/request_forms_and_files/tutorial001_an_py310.py hl[10:12] *}

The files and form fields will be uploaded as form data and you will receive the files and form fields.

And you can declare some of the files as `bytes` and some as `UploadFile`.

/// warning

You can declare multiple `File` and `Form` parameters in a *path operation*, but you can't also declare `Body` fields that you expect to receive as JSON, as the request will have the body encoded using `multipart/form-data` instead of `application/json`.

This is not a limitation of **FastAPI**, it's part of the HTTP protocol.

///

</details>

<details>
<summary>Source 2: tutorial/request-files.md (Request Files { #request-files } > What is "Form Data" { #what-is-form-data })</summary>

The way HTML forms (`<form></form>`) sends the data to the server normally uses a "special" encoding for that data, it's different from JSON.

**FastAPI** will make sure to read that data from the right place instead of JSON.

/// note | Technical Details

Data from forms is normally encoded using the "media type" `application/x-www-form-urlencoded` when it doesn't include files.

But when the form includes files, it is encoded as `multipart/form-data`. If you use `File`, **FastAPI** will know it has to get the files from the correct part of the body.

If you want to read more about these encodings and form fields, head to the [<abbr title="Mozilla Developer Network">MDN</abbr> web docs for `POST`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST).

///

/// warning

You can declare multiple `File` and `Form` parameters in a *path operation*, but you can't also declare `Body` fields that you expect to receive as JSON, as the request will have the body encoded using `multipart/form-data` instead of `application/json`.

This is not a limitation of **FastAPI**, it's part of the HTTP protocol.

///

</details>

<details>
<summary>Source 3: tutorial/request-forms-and-files.md (Request Forms and Files { #request-forms-and-files } > Recap { #recap })</summary>

Use `File` and `Form` together when you need to receive data and files in the same request.

</details>

<details>
<summary>Source 4: tutorial/request-form-models.md (Form Models { #form-models } > Pydantic Models for Forms { #pydantic-models-for-forms })</summary>

You just need to declare a **Pydantic model** with the fields you want to receive as **form fields**, and then declare the parameter as `Form`:

{* ../../docs_src/request_form_models/tutorial001_an_py310.py hl[9:11,15] *}

**FastAPI** will **extract** the data for **each field** from the **form data** in the request and give you the Pydantic model you defined.

</details>

<details>
<summary>Source 5: tutorial/request-forms.md (Form Data { #form-data } > About "Form Fields" { #about-form-fields })</summary>

The way HTML forms (`<form></form>`) sends the data to the server normally uses a "special" encoding for that data, it's different from JSON.

**FastAPI** will make sure to read that data from the right place instead of JSON.

/// note | Technical Details

Data from forms is normally encoded using the "media type" `application/x-www-form-urlencoded`.

But when the form includes files, it is encoded as `multipart/form-data`. You'll read about handling files in the next chapter.

If you want to read more about these encodings and form fields, head to the [<abbr title="Mozilla Developer Network">MDN</abbr> web docs for `POST`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST).

///

/// warning

You can declare multiple `Form` parameters in a *path operation*, but you can't also declare `Body` fields that you expect to receive as JSON, as the request will have the body encoded using `application/x-www-form-urlencoded` instead of `application/json`.

This is not a limitation of **FastAPI**, it's part of the HTTP protocol.

///

</details>

**Answer:**

Based on the provided context, here’s how you can get files and form data using the Request object in FastAPI:

1.  **Use `File` and `Form` parameters:** You can declare multiple `File` and `Form` parameters in a *path operation*. These will be uploaded as form data.
2.  **Pydantic Models:** You just need to declare a **Pydantic model** with the fields you want to receive as **form fields**, and then declare the parameter as `Form`. FastAPI will then **extract** the data for **each field** from the **form data** in the request and give you the Pydantic model you defined.
3.  **Encoding:**  HTML forms use a "special" encoding (different from JSON) when sending data. FastAPI ensures it reads the data from the correct place, using `application/x-www-form-urlencoded` for form data without files and `multipart/form-data` when files are included.

You can find more information about these encodings and form fields in the MDN web docs for `POST`: [https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST)

---

## j-25: How to add drop down menu to Swagger UI autodocs based on BaseModel using FastAPI?

Question: [so_74366289](https://stackoverflow.com/questions/74366289/how-to-add-drop-down-menu-to-swagger-ui-autodocs-based-on-basemodel-using-fastap) (CC BY-SA 4.0)

<details>
<summary>Source 1: how-to/configure-swagger-ui.md (Configure Swagger UI { #configure-swagger-ui })</summary>

You can configure some extra [Swagger UI parameters](https://swagger.io/docs/open-source-tools/swagger-ui/usage/configuration/).

To configure them, pass the `swagger_ui_parameters` argument when creating the `FastAPI()` app object or to the `get_swagger_ui_html()` function.

`swagger_ui_parameters` receives a dictionary with the configurations passed to Swagger UI directly.

FastAPI converts the configurations to **JSON** to make them compatible with JavaScript, as that's what Swagger UI needs.

</details>

<details>
<summary>Source 2: tutorial/first-steps.md (First Steps { #first-steps } > Check it { #check-it } > Interactive API docs { #interactive-api-docs })</summary>

Now go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

You will see the automatic interactive API documentation (provided by [Swagger UI](https://github.com/swagger-api/swagger-ui)):

![Swagger UI](https://fastapi.tiangolo.com/img/index/index-01-swagger-ui-simple.png)

</details>

<details>
<summary>Source 3: release-notes.md (Release Notes > 0.104.1 (2023-10-30))</summary>

📌 Pin Swagger UI version to 5.9.0 temporarily to handle a bug crashing it in 5.9.1. PR [#10529](https://github.com/tiangolo/fastapi/pull/10529) by [@alejandraklachquin](https://github.com/alejandraklachquin).

This is not really a bug in FastAPI but in Swagger UI, nevertheless pinning the version will work while a solution is found on the [Swagger UI side](https://github.com/swagger-api/swagger-ui/issues/9337).

📝 Update data structure and render for external-links. PR [#10495](https://github.com/tiangolo/fastapi/pull/10495) by [@tiangolo](https://github.com/tiangolo).

✏️ Fix link to SPDX license identifier in `docs/en/docs/tutorial/metadata.md`. PR [#10433](https://github.com/tiangolo/fastapi/pull/10433) by [@worldworm](https://github.com/worldworm).

📝 Update example validation error from Pydantic v1 to match Pydantic v2 in `docs/en/docs/tutorial/path-params.md`. PR [#10043](https://github.com/tiangolo/fastapi/pull/10043) by [@giuliowaitforitdavide](https://github.com/giuliowaitforitdavide).

✏️ Fix typos in emoji docs and in some source examples. PR [#10438](https://github.com/tiangolo/fastapi/pull/10438) by [@afuetterer](https://github.com/afuetterer).

✏️ Fix typo in `docs/en/docs/reference/dependencies.md`. PR [#10465](https://github.com/tiangolo/fastapi/pull/10465) by [@suravshresth](https://github.com/suravshresth).

✏️ Fix typos and rewordings in `docs/en/docs/tutorial/body-nested-models.md`. PR [#10468](https://github.com/tiangolo/fastapi/pull/10468) by [@yogabonito](https://github.com/yogabonito).

📝 Update docs, remove references to removed `pydantic.Required` in `docs/en/docs/tutorial/query-params-str-validations.md`. PR [#10469](https://github.com/tiangolo/fastapi/pull/10469) by [@yogabonito](https://github.com/yogabonito).

✏️ Fix typo in `docs/en/docs/reference/index.md`. PR [#10467](https://github.com/tiangolo/fastapi/pull/10467) by [@tarsil](https://github.com/tarsil).

🔥 Remove unnecessary duplicated docstrings. PR [#10484](https://github.com/tiangolo/fastapi/pull/10484) by [@tiangolo](https://github.com/tiangolo).

✏️ Update Pydantic links to dotenv support. PR [#10511](https://github.com/tiangolo/fastapi/pull/10511) by [@White-Mask](https://github.com/White-Mask).

✏️ Update links in `docs/en/docs/async.md` and `docs/zh/docs/async.md` to make them relative. PR [#10498](https://github.com/tiangolo/fastapi/pull/10498) by [@hasnatsajid](https://github.com/hasnatsajid).

✏️ Fix links in `docs/em/docs/async.md`. PR [#10507](https://github.com/tiangolo/fastapi/pull/10507) by [@hasnatsajid](https://github.com/hasnatsajid).

✏️ Fix typo in `docs/em/docs/index.md`, Python 3.8. PR [#10521](https://github.com/tiangolo/fastapi/pull/10521) by [@kerriop](https://github.com/kerriop).

⬆ Bump pillow from 9.5.0 to 10.1.0. PR [#10446](https://github.com/tiangolo/fastapi/pull/10446) by [@dependabot[bot]](https://github.com/apps/dependabot).

⬆ Update mkdocs-material requirement from <9.0.0,>=8.1.4 to >=8.1.4,<10.0.0. PR [#5862](https://github.com/tiangolo/fastapi/pull/5862) by [@dependabot[bot]](https://github.com/apps/dependabot).

⬆ Bump mkdocs-material from 9.1.21 to 9.4.7. PR [#10545](https://github.com/tiangolo/fastapi/pull/10545) by [@dependabot[bot]](https://github.com/apps/dependabot).

👷 Install MkDocs Material Insiders only when secrets are available, for Dependabot. PR [#10544](https://github.com/tiangolo/fastapi/pull/10544) by [@tiangolo](https://github.com/tiangolo).

🔧 Update sponsors badges, Databento. PR [#10519](https://github.com/tiangolo/fastapi/pull/10519) by [@tiangolo](https://github.com/tiangolo).

👷 Adopt Ruff format. PR [#10517](https://github.com/tiangolo/fastapi/pull/10517) by [@tiangolo](https://github.com/tiangolo).

🔧 Add `CITATION.cff` file for academic citations. PR [#10496](https://github.com/tiangolo/fastapi/pull/10496) by [@tiangolo](https://github.com/tiangolo).

🐛 Fix overriding MKDocs theme lang in hook. PR [#10490](https://github.com/tiangolo/fastapi/pull/10490) by [@tiangolo](https://github.com/tiangolo).

🔥 Drop/close Gitter chat. Questions should go to GitHub Discussions, free conversations to Discord.. PR [#10485](https://github.com/tiangolo/fastapi/pull/10485) by [@tiangolo](https://github.com/tiangolo).

</details>

<details>
<summary>Source 4: features.md (Features { #features } > FastAPI features { #fastapi-features } > Automatic docs { #automatic-docs })</summary>

Interactive API documentation and exploration web user interfaces. As the framework is based on OpenAPI, there are multiple options, 2 included by default.

[**Swagger UI**](https://github.com/swagger-api/swagger-ui), with interactive exploration, call and test your API directly from the browser.

![Swagger UI interaction](https://fastapi.tiangolo.com/img/index/index-03-swagger-02.png)

Alternative API documentation with [**ReDoc**](https://github.com/Rebilly/ReDoc).

![ReDoc](https://fastapi.tiangolo.com/img/index/index-06-redoc-02.png)

</details>

<details>
<summary>Source 5: deployment/docker.md (FastAPI in Containers - Docker { #fastapi-in-containers-docker } > Interactive API docs { #interactive-api-docs })</summary>

Now you can go to [http://192.168.99.100/docs](http://192.168.99.100/docs) or [http://127.0.0.1/docs](http://127.0.0.1/docs) (or equivalent, using your Docker host).

You will see the automatic interactive API documentation (provided by [Swagger UI](https://github.com/swagger-api/swagger-ui)):

![Swagger UI](https://fastapi.tiangolo.com/img/index/index-01-swagger-ui-simple.png)

</details>

**Answer:**

The provided context doesn't contain information on how to add a dropdown menu to Swagger UI based on BaseModels. 

However, it does provide information on configuring Swagger UI parameters. You can pass the `swagger_ui_parameters` argument when creating the `FastAPI()` app object or to the `get_swagger_ui_html()` function. This dictionary can be used to configure Swagger UI directly.  It also mentions that FastAPI converts the configurations to JSON to make them compatible with JavaScript, as that's what Swagger UI needs.

Additionally, the context discusses the use of Swagger UI and Redoc as interactive API documentation tools. It provides links to their respective GitHub repositories and example images. 

To get a dropdown menu based on BaseModels, you would likely need to explore advanced Swagger UI configuration options, potentially using JavaScript or a plugin to extend Swagger UI's functionality. The documentation for Swagger UI itself would be the best resource for this.

---

## j-26: How to save UploadFile in FastAPI

Question: [so_63580229](https://stackoverflow.com/questions/63580229/how-to-save-uploadfile-in-fastapi) (CC BY-SA 4.0)

<details>
<summary>Source 1: reference/uploadfile.md (`UploadFile` class)</summary>

You can define *path operation function* parameters to be of the type `UploadFile` to receive files from the request.

You can import it directly from `fastapi`:


```python
from fastapi import UploadFile
```
::: fastapi.UploadFile
    options:
        members:
            - file
            - filename
            - size
            - headers
            - content_type
            - read
            - write
            - seek
            - close

</details>

<details>
<summary>Source 2: tutorial/request-files.md (Request Files { #request-files } > Multiple File Uploads { #multiple-file-uploads })</summary>

It's possible to upload several files at the same time.

They would be associated to the same "form field" sent using "form data".

To use that, declare a list of `bytes` or `UploadFile`:

{* ../../docs_src/request_files/tutorial002_an_py310.py hl[10,15] *}

You will receive, as declared, a `list` of `bytes` or `UploadFile`s.

/// note | Technical Details

You could also use `from starlette.responses import HTMLResponse`.

**FastAPI** provides the same `starlette.responses` as `fastapi.responses` just as a convenience for you, the developer. But most of the available responses come directly from Starlette.

///

And the same way as before, you can use `File()` to set additional parameters, even for `UploadFile`:

{* ../../docs_src/request_files/tutorial003_an_py310.py hl[11,18:20] *}

</details>

<details>
<summary>Source 3: release-notes.md (Release Notes > 0.7.0 (2019-03-03))</summary>

Add support for `UploadFile` in `File` parameter annotations.

This includes a file-like interface.

Here's the updated documentation for declaring [`File` parameters with `UploadFile`](https://fastapi.tiangolo.com/tutorial/request-files/#file-parameters-with-uploadfile).

And here's the updated documentation for using [`Form` parameters mixed with `File` parameters, supporting `bytes` and `UploadFile`](https://fastapi.tiangolo.com/tutorial/request-forms-and-files/) at the same time.

PR [#63](https://github.com/tiangolo/fastapi/pull/63).

</details>

<details>
<summary>Source 4: tutorial/request-files.md (Request Files { #request-files } > Recap { #recap })</summary>

Use `File`, `bytes`, and `UploadFile` to declare files to be uploaded in the request, sent as form data.

</details>

<details>
<summary>Source 5: tutorial/request-files.md (Request Files { #request-files } > Import `File` { #import-file })</summary>

Import `File` and `UploadFile` from `fastapi`:

{* ../../docs_src/request_files/tutorial001_an_py310.py hl[3] *}

</details>

**Answer:**

According to the documentation, you can use `UploadFile` to receive files from the request. You can import it directly from `fastapi`:

```python
from fastapi import UploadFile
```

You can also upload several files at the same time using a list of `bytes` or `UploadFile`s, associated to the same "form field" sent using "form data".  The `File()` parameter annotation now supports `UploadFile`.  You can use `File`, `bytes`, and `UploadFile` to declare files to be uploaded in the request, sent as form data.

---

## j-27: Overriding FastAPI's HTTPException response body

Question: [so_66273149](https://stackoverflow.com/questions/66273149/overriding-fastapis-httpexception-response-body) (CC BY-SA 4.0)

<details>
<summary>Source 1: tutorial/handling-errors.md (Handling Errors { #handling-errors } > Override the default exception handlers { #override-the-default-exception-handlers })</summary>

**FastAPI** has some default exception handlers.

These handlers are in charge of returning the default JSON responses when you `raise` an `HTTPException` and when the request has invalid data.

You can override these exception handlers with your own.

When a request contains invalid data, **FastAPI** internally raises a `RequestValidationError`.

And it also includes a default exception handler for it.

To override it, import the `RequestValidationError` and use it with `@app.exception_handler(RequestValidationError)` to decorate the exception handler.

The exception handler will receive a `Request` and the exception.

{* ../../docs_src/handling_errors/tutorial004_py310.py hl[2,14:19] *}

Now, if you go to `/items/foo`, instead of getting the default JSON error with:


```JSON
{
    "detail": [
        {
            "loc": [
                "path",
                "item_id"
            ],
            "msg": "value is not a valid integer",
            "type": "type_error.integer"
        }
    ]
}
```
you will get a text version, with:


```
Validation errors:
Field: ('path', 'item_id'), Error: Input should be a valid integer, unable to parse string as an integer
```

The same way, you can override the `HTTPException` handler.

For example, you could want to return a plain text response instead of JSON for these errors:

{* ../../docs_src/handling_errors/tutorial004_py310.py hl[3:4,9:11,25] *}

/// note | Technical Details

You could also use `from starlette.responses import PlainTextResponse`.

**FastAPI** provides the same `starlette.responses` as `fastapi.responses` just as a convenience for you, the developer. But most of the available responses come directly from Starlette.

///

/// warning

Have in mind that the `RequestValidationError` contains the information of the file name and line where the validation error happens so that you can show it in your logs with the relevant information if you want to.

But that means that if you just convert it to a string and return that information directly, you could be leaking a bit of information about your system, that's why here the code extracts and shows each error independently.

///

The `RequestValidationError` contains the `body` it received with invalid data.

You could use it while developing your app to log the body and debug it, return it to the user, etc.

{* ../../docs_src/handling_errors/tutorial005_py310.py hl[14] *}

Now try sending an invalid item like:


```JSON
{
  "title": "towel",
  "size": "XL"
}
```
You will receive a response telling you that the data is invalid containing the received body:


```JSON hl_lines="12-15"
{
  "detail": [
    {
      "loc": [
        "body",
        "size"
      ],
      "msg": "value is not a valid integer",
      "type": "type_error.integer"
    }
  ],
  "body": {
    "title": "towel",
    "size": "XL"
  }
}
```

**FastAPI** has its own `HTTPException`.

And **FastAPI**'s `HTTPException` error class inherits from Starlette's `HTTPException` error class.

The only difference is that **FastAPI**'s `HTTPException` accepts any JSON-able data for the `detail` field, while Starlette's `HTTPException` only accepts strings for it.

So, you can keep raising **FastAPI**'s `HTTPException` as normally in your code.

But when you register an exception handler, you should register it for Starlette's `HTTPException`.

This way, if any part of Starlette's internal code, or a Starlette extension or plug-in, raises a Starlette `HTTPException`, your handler will be able to catch and handle it.

In this example, to be able to have both `HTTPException`s in the same code, Starlette's exceptions is renamed to `StarletteHTTPException`:


```Python
from starlette.exceptions import HTTPException as StarletteHTTPException
```

If you want to use the exception along with the same default exception handlers from  **FastAPI**, you can import and reuse the default exception handlers from `fastapi.exception_handlers`:

{* ../../docs_src/handling_errors/tutorial006_py310.py hl[2:5,15,21] *}

In this example you are just printing the error with a very expressive message, but you get the idea. You can use the exception and then just reuse the default exception handlers.

</details>

<details>
<summary>Source 2: reference/responses.md (Custom Response Classes - File, HTML, Redirect, Streaming, etc. > Starlette Responses)</summary>

You can read more about all of them in the [FastAPI docs for Custom Response](https://fastapi.tiangolo.com/advanced/custom-response/) and in the [Starlette docs about Responses](https://starlette.dev/responses/).

::: fastapi.responses.FileResponse
    options:
        members:
            - chunk_size
            - charset
            - status_code
            - media_type
            - body
            - background
            - raw_headers
            - render
            - init_headers
            - headers
            - set_cookie
            - delete_cookie

::: fastapi.responses.HTMLResponse
    options:
        members:
            - charset
            - status_code
            - media_type
            - body
            - background
            - raw_headers
            - render
            - init_headers
            - headers
            - set_cookie
            - delete_cookie

::: fastapi.responses.JSONResponse
    options:
        members:
            - charset
            - status_code
            - media_type
            - body
            - background
            - raw_headers
            - render
            - init_headers
            - headers
            - set_cookie
            - delete_cookie

::: fastapi.responses.PlainTextResponse
    options:
        members:
            - charset
            - status_code
            - media_type
            - body
            - background
            - raw_headers
            - render
            - init_headers
            - headers
            - set_cookie
            - delete_cookie

::: fastapi.responses.RedirectResponse
    options:
        members:
            - charset
            - status_code
            - media_type
            - body
            - background
            - raw_headers
            - render
            - init_headers
            - headers
            - set_cookie
            - delete_cookie

::: fastapi.responses.Response
    options:
        members:
            - charset
            - status_code
            - media_type
            - body
            - background
            - raw_headers
            - render
            - init_headers
            - headers
            - set_cookie
            - delete_cookie

::: fastapi.responses.StreamingResponse
    options:
        members:
            - body_iterator
            - charset
            - status_code
            - media_type
            - body
            - background
            - raw_headers
            - render
            - init_headers
            - headers
            - set_cookie
            - delete_cookie

</details>

<details>
<summary>Source 3: reference/security/index.md (Security Tools > HTTP Authentication Schemes)</summary>

::: fastapi.security.HTTPBasic

::: fastapi.security.HTTPBearer

::: fastapi.security.HTTPDigest

</details>

<details>
<summary>Source 4: advanced/custom-response.md (Custom Response - HTML, Stream, File, others { #custom-response-html-stream-file-others })</summary>

By default, **FastAPI** will return JSON responses.

You can override it by returning a `Response` directly as seen in [Return a Response directly](response-directly.md).

But if you return a `Response` directly (or any subclass, like `JSONResponse`), the data won't be automatically converted (even if you declare a `response_model`), and the documentation won't be automatically generated (for example, including the specific "media type", in the HTTP header `Content-Type` as part of the generated OpenAPI).

But you can also declare the `Response` that you want to be used (e.g. any `Response` subclass), in the *path operation decorator* using the `response_class` parameter.

The contents that you return from your *path operation function* will be put inside of that `Response`.

/// note

If you use a response class with no media type, FastAPI will expect your response to have no content, so it will not document the response format in its generated OpenAPI docs.

///

</details>

<details>
<summary>Source 5: tutorial/handling-errors.md (Handling Errors { #handling-errors } > Override the default exception handlers { #override-the-default-exception-handlers })</summary>

**FastAPI** has some default exception handlers.

These handlers are in charge of returning the default JSON responses when you `raise` an `HTTPException` and when the request has invalid data.

You can override these exception handlers with your own.

When a request contains invalid data, **FastAPI** internally raises a `RequestValidationError`.

And it also includes a default exception handler for it.

To override it, import the `RequestValidationError` and use it with `@app.exception_handler(RequestValidationError)` to decorate the exception handler.

The exception handler will receive a `Request` and the exception.

{* ../../docs_src/handling_errors/tutorial004_py310.py hl[2,14:19] *}

Now, if you go to `/items/foo`, instead of getting the default JSON error with:


```JSON
{
    "detail": [
        {
            "loc": [
                "path",
                "item_id"
            ],
            "msg": "value is not a valid integer",
            "type": "type_error.integer"
        }
    ]
}
```
you will get a text version, with:


```
Validation errors:
Field: ('path', 'item_id'), Error: Input should be a valid integer, unable to parse string as an integer
```

The same way, you can override the `HTTPException` handler.

For example, you could want to return a plain text response instead of JSON for these errors:

{* ../../docs_src/handling_errors/tutorial004_py310.py hl[3:4,9:11,25] *}

/// note | Technical Details

You could also use `from starlette.responses import PlainTextResponse`.

**FastAPI** provides the same `starlette.responses` as `fastapi.responses` just as a convenience for you, the developer. But most of the available responses come directly from Starlette.

///

/// warning

Have in mind that the `RequestValidationError` contains the information of the file name and line where the validation error happens so that you can show it in your logs with the relevant information if you want to.

But that means that if you just convert it to a string and return that information directly, you could be leaking a bit of information about your system, that's why here the code extracts and shows each error independently.

///

The `RequestValidationError` contains the `body` it received with invalid data.

You could use it while developing your app to log the body and debug it, return it to the user, etc.

{* ../../docs_src/handling_errors/tutorial005_py310.py hl[14] *}

Now try sending an invalid item like:


```JSON
{
  "title": "towel",
  "size": "XL"
}
```
You will receive a response telling you that the data is invalid containing the received body:


```JSON hl_lines="12-15"
{
  "detail": [
    {
      "loc": [
        "body",
        "size"
      ],
      "msg": "value is not a valid integer",
      "type": "type_error.integer"
    }
  ],
  "body": {
    "title": "towel",
    "size": "XL"
  }
}
```

**FastAPI** has its own `HTTPException`.

And **FastAPI**'s `HTTPException` error class inherits from Starlette's `HTTPException` error class.

The only difference is that **FastAPI**'s `HTTPException` accepts any JSON-able data for the `detail` field, while Starlette's `HTTPException` only accepts strings for it.

So, you can keep raising **FastAPI**'s `HTTPException` as normally in your code.

But when you register an exception handler, you should register it for Starlette's `HTTPException`.

This way, if any part of Starlette's internal code, or a Starlette extension or plug-in, raises a Starlette `HTTPException`, your handler will be able to catch and handle it.

In this example, to be able to have both `HTTPException`s in the same code, Starlette's exceptions is renamed to `StarletteHTTPException`:


```Python
from starlette.exceptions import HTTPException as StarletteHTTPException
```

If you want to use the exception along with the same default exception handlers from  **FastAPI**, you can import and reuse the default exception handlers from `fastapi.exception_handlers`:

{* ../../docs_src/handling_errors/tutorial006_py310.py hl[2:5,15,21] *}

In this example you are just printing the error with a very expressive message, but you get the idea. You can use the exception and then just reuse the default exception handlers.

</details>

**Answer:**

FastAPI has default exception handlers that return JSON responses by default. You can override these handlers to return different response bodies, such as plain text.

Specifically, you can override the `HTTPException` handler using `@app.exception_handler(RequestValidationError)`. This handler receives a `Request` and the exception. To return a plain text response instead of JSON, you can import and reuse the default exception handlers from `fastapi.exception_handlers`.

When a request contains invalid data, FastAPI internally raises a `RequestValidationError`. You can override this handler to customize the error response. The `RequestValidationError` contains the body of the invalid data, which can be useful for logging and debugging.

---

## j-28: Can't access path parameters from middleware

Question: [so_62895883](https://stackoverflow.com/questions/62895883/cant-access-path-parameters-from-middleware) (CC BY-SA 4.0)

<details>
<summary>Source 1: tutorial/query-params.md (Query Parameters { #query-parameters } > Multiple path and query parameters { #multiple-path-and-query-parameters })</summary>

You can declare multiple path parameters and query parameters at the same time, **FastAPI** knows which is which.

And you don't have to declare them in any specific order.

They will be detected by name:

{* ../../docs_src/query_params/tutorial004_py310.py hl[6,8] *}

</details>

<details>
<summary>Source 2: tutorial/body.md (Request Body { #request-body } > Request body + path parameters { #request-body-path-parameters })</summary>

You can declare path parameters and request body at the same time.

**FastAPI** will recognize that the function parameters that match path parameters should be **taken from the path**, and that function parameters that are declared to be Pydantic models should be **taken from the request body**.

{* ../../docs_src/body/tutorial003_py310.py hl[15:16] *}

</details>

<details>
<summary>Source 3: tutorial/path-operation-configuration.md (Path Operation Configuration { #path-operation-configuration })</summary>

There are several parameters that you can pass to your *path operation decorator* to configure it.

/// warning

Notice that these parameters are passed directly to the *path operation decorator*, not to your *path operation function*.

///

</details>

<details>
<summary>Source 4: tutorial/path-params-numeric-validations.md (Path Parameters and Numeric Validations { #path-parameters-and-numeric-validations })</summary>

In the same way that you can declare more validations and metadata for query parameters with `Query`, you can declare the same type of validations and metadata for path parameters with `Path`.

</details>

<details>
<summary>Source 5: tutorial/cookie-params.md (Cookie Parameters { #cookie-parameters })</summary>

You can define Cookie parameters the same way you define `Query` and `Path` parameters.

</details>

**Answer:**

The provided context doesn't directly address the issue of accessing path parameters from middleware. However, it does explain how FastAPI handles path parameters, query parameters, and request bodies in relation to path operation functions. 

Specifically, it states that FastAPI detects parameters by name and that path parameters are taken from the path, while Pydantic models are taken from the request body. 

To help you further, I would need more information about your middleware and how you are attempting to access the path parameters within it.

---

## j-29: How to exclude Optional unset values from a Pydantic model using FastAPI?

Question: [so_73198957](https://stackoverflow.com/questions/73198957/how-to-exclude-optional-unset-values-from-a-pydantic-model-using-fastapi) (CC BY-SA 4.0)

<details>
<summary>Source 1: tutorial/response-model.md (Response Model - Return Type { #response-model-return-type } > Response Model encoding parameters { #response-model-encoding-parameters })</summary>

Your response model could have default values, like:

{* ../../docs_src/response_model/tutorial004_py310.py hl[9,11:12] *}

`description: Union[str, None] = None` (or `str | None = None` in Python 3.10) has a default of `None`.

`tax: float = 10.5` has a default of `10.5`.

`tags: List[str] = []` has a default of an empty list: `[]`.

but you might want to omit them from the result if they were not actually stored.

For example, if you have models with many optional attributes in a NoSQL database, but you don't want to send very long JSON responses full of default values.

You can set the *path operation decorator* parameter `response_model_exclude_unset=True`:

{* ../../docs_src/response_model/tutorial004_py310.py hl[22] *}

and those default values won't be included in the response, only the values actually set.

So, if you send a request to that *path operation* for the item with ID `foo`, the response (not including default values) will be:


```JSON
{
    "name": "Foo",
    "price": 50.2
}
```
/// info

You can also use:

`response_model_exclude_defaults=True`

`response_model_exclude_none=True`

as described in [the Pydantic docs](https://docs.pydantic.dev/1.10/usage/exporting_models/#modeldict) for `exclude_defaults` and `exclude_none`.

///

But if your data has values for the model's fields with default values, like the item with ID `bar`:


```Python hl_lines="3  5"
{
    "name": "Bar",
    "description": "The bartenders",
    "price": 62,
    "tax": 20.2
}
```
they will be included in the response.

If the data has the same values as the default ones, like the item with ID `baz`:


```Python hl_lines="3  5-6"
{
    "name": "Baz",
    "description": None,
    "price": 50.2,
    "tax": 10.5,
    "tags": []
}
```
FastAPI is smart enough (actually, Pydantic is smart enough) to realize that, even though `description`, `tax`, and `tags` have the same values as the defaults, they were set explicitly (instead of taken from the defaults).

So, they will be included in the JSON response.

/// tip

Notice that the default values can be anything, not only `None`.

They can be a list (`[]`), a `float` of `10.5`, etc.

///

You can also use the *path operation decorator* parameters `response_model_include` and `response_model_exclude`.

They take a `set` of `str` with the name of the attributes to include (omitting the rest) or to exclude (including the rest).

This can be used as a quick shortcut if you have only one Pydantic model and want to remove some data from the output.

/// tip

But it is still recommended to use the ideas above, using multiple classes, instead of these parameters.

This is because the JSON Schema generated in your app's OpenAPI (and the docs) will still be the one for the complete model, even if you use `response_model_include` or `response_model_exclude` to omit some attributes.

This also applies to `response_model_by_alias` that works similarly.

///

{* ../../docs_src/response_model/tutorial005_py310.py hl[29,35] *}

/// tip

The syntax `{"name", "description"}` creates a `set` with those two values.

It is equivalent to `set(["name", "description"])`.

///

If you forget to use a `set` and use a `list` or `tuple` instead, FastAPI will still convert it to a `set` and it will work correctly:

{* ../../docs_src/response_model/tutorial006_py310.py hl[29,35] *}

</details>

<details>
<summary>Source 2: release-notes.md (Release Notes > 0.61.0 (2020-08-09))</summary>

Add support for injecting `HTTPConnection` (as `Request` and `WebSocket`). Useful for sharing app state in dependencies. PR [#1827](https://github.com/tiangolo/fastapi/pull/1827) by [@nsidnev](https://github.com/nsidnev).

Export `WebSocketDisconnect` and add example handling WebSocket disconnections to docs. PR [#1822](https://github.com/tiangolo/fastapi/pull/1822) by [@rkbeatss](https://github.com/rkbeatss).

Require Pydantic > `1.0.0`.

Remove support for deprecated Pydantic `0.32.2`. This improves maintainability and allows new features.

In `FastAPI` and `APIRouter`:

Remove *path operation decorators* related/deprecated parameter `response_model_skip_defaults` (use `response_model_exclude_unset` instead).

Change *path operation decorators* parameter default for `response_model_exclude` from `set()` to `None` (as is in Pydantic).

In `encoders.jsonable_encoder`:

Remove deprecated `skip_defaults`, use instead `exclude_unset`.

Set default of `exclude` from `set()` to `None` (as is in Pydantic).

PR [#1862](https://github.com/tiangolo/fastapi/pull/1862).

In `encoders.jsonable_encoder` remove parameter `sqlalchemy_safe`.

It was an early hack to allow returning SQLAlchemy models, but it was never documented, and the recommended way is using Pydantic's `orm_mode` as described in the tutorial: [SQL (Relational) Databases](https://fastapi.tiangolo.com/tutorial/sql-databases/).

PR [#1864](https://github.com/tiangolo/fastapi/pull/1864).

Add link to the course by TestDriven.io: [Test-Driven Development with FastAPI and Docker](https://testdriven.io/courses/tdd-fastapi/). PR [#1860](https://github.com/tiangolo/fastapi/pull/1860).

Fix empty log message in docs example about handling errors. PR [#1815](https://github.com/tiangolo/fastapi/pull/1815) by [@manlix](https://github.com/manlix).

Reword text to reduce ambiguity while not being gender-specific. PR [#1824](https://github.com/tiangolo/fastapi/pull/1824) by [@Mause](https://github.com/Mause).

Add Flake8 linting. Original PR [#1774](https://github.com/tiangolo/fastapi/pull/1774) by [@MashhadiNima](https://github.com/MashhadiNima).

Disable Gitter bot, as it's currently broken, and Gitter's response doesn't show the problem. PR [#1853](https://github.com/tiangolo/fastapi/pull/1853).

</details>

<details>
<summary>Source 3: release-notes.md (Release Notes > 0.25.0 (2019-05-27))</summary>

Add support for Pydantic's `include`, `exclude`, `by_alias`.

Update documentation: [Response Model](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).

Add docs for: [Body - updates](https://fastapi.tiangolo.com/tutorial/body-updates/), using Pydantic's `skip_defaults`.

Add method consistency tests.

PR [#264](https://github.com/tiangolo/fastapi/pull/264).

Add `CONTRIBUTING.md` file to GitHub, to help new contributors. PR [#255](https://github.com/tiangolo/fastapi/pull/255) by [@wshayes](https://github.com/wshayes).

Add support for Pydantic's `skip_defaults`:

There's a new *path operation decorator* parameter `response_model_skip_defaults`.

The name of the parameter will most probably change in a future version to `response_skip_defaults`, `model_skip_defaults` or something similar.

New [documentation section about using `response_model_skip_defaults`](https://fastapi.tiangolo.com/tutorial/response-model/#response-model-encoding-parameters).

PR [#248](https://github.com/tiangolo/fastapi/pull/248) by [@wshayes](https://github.com/wshayes).

</details>

<details>
<summary>Source 4: release-notes.md (Release Notes > 0.119.0 (2025-10-11))</summary>

FastAPI now (temporarily) supports both Pydantic v2 models and `pydantic.v1` models at the same time in the same app, to make it easier for any FastAPI apps still using Pydantic v1 to gradually but quickly **migrate to Pydantic v2**.


```Python
from fastapi import FastAPI
from pydantic import BaseModel as BaseModelV2
from pydantic.v1 import BaseModel


class Item(BaseModel):
    name: str
    description: str | None = None


class ItemV2(BaseModelV2):
    title: str
    summary: str | None = None


app = FastAPI()


@app.post("/items/", response_model=ItemV2)
def create_item(item: Item):
    return {"title": item.name, "summary": item.description}
```
Adding this feature was a big effort with the main objective of making it easier for the few applications still stuck in Pydantic v1 to migrate to Pydantic v2.

And with this, support for **Pydantic v1 is now deprecated** and will be **removed** from FastAPI in a future version soon.

**Note**: have in mind that the Pydantic team already stopped supporting Pydantic v1 for recent versions of Python, starting with Python 3.14.

You can read in the docs more about how to [Migrate from Pydantic v1 to Pydantic v2](https://fastapi.tiangolo.com/how-to/migrate-from-pydantic-v1-to-pydantic-v2/).

✨ Add support for `from pydantic.v1 import BaseModel`, mixed Pydantic v1 and v2 models in the same app. PR [#14168](https://github.com/fastapi/fastapi/pull/14168) by [@tiangolo](https://github.com/tiangolo).

</details>

<details>
<summary>Source 5: advanced/dataclasses.md (Using Dataclasses { #using-dataclasses })</summary>

FastAPI is built on top of **Pydantic**, and I have been showing you how to use Pydantic models to declare requests and responses.

But FastAPI also supports using [`dataclasses`](https://docs.python.org/3/library/dataclasses.html) the same way:

{* ../../docs_src/dataclasses_/tutorial001_py310.py hl[1,6:11,18:19] *}

This is still supported thanks to **Pydantic**, as it has [internal support for `dataclasses`](https://docs.pydantic.dev/latest/concepts/dataclasses/#use-of-stdlib-dataclasses-with-basemodel).

So, even with the code above that doesn't use Pydantic explicitly, FastAPI is using Pydantic to convert those standard dataclasses to Pydantic's own flavor of dataclasses.

And of course, it supports the same:

data validation

data serialization

data documentation, etc.

This works the same way as with Pydantic models. And it is actually achieved in the same way underneath, using Pydantic.

/// info

Keep in mind that dataclasses can't do everything Pydantic models can do.

So, you might still need to use Pydantic models.

But if you have a bunch of dataclasses laying around, this is a nice trick to use them to power a web API using FastAPI. 🤓

///

</details>

**Answer:**

You can use the `response_model_exclude_unset=True` parameter on the *path operation decorator* to exclude optional unset values from the response. This means that only the values that are actually set in the data will be included in the JSON response.

For example, if you have a model with optional attributes and you set `response_model_exclude_unset=True`, then if you send a request to that *path operation* for the item with ID `foo`, the response (not including default values) will be:

```JSON
{
    "name": "Foo",
    "price": 50.2
}
```

Alternatively, you can use `response_model_exclude_defaults=True` or `response_model_exclude_none=True` to exclude default values or None values respectively.  Also, you can use `response_model_include` and `response_model_exclude` to specify the attributes to include or exclude from the response.

---

## j-30: Query parameters from pydantic model

Question: [so_62468402](https://stackoverflow.com/questions/62468402/query-parameters-from-pydantic-model) (CC BY-SA 4.0)

<details>
<summary>Source 1: tutorial/query-param-models.md (Query Parameter Models { #query-parameter-models } > Query Parameters with a Pydantic Model { #query-parameters-with-a-pydantic-model })</summary>

Declare the **query parameters** that you need in a **Pydantic model**, and then declare the parameter as `Query`:

{* ../../docs_src/query_param_models/tutorial001_an_py310.py hl[9:13,17] *}

**FastAPI** will **extract** the data for **each field** from the **query parameters** in the request and give you the Pydantic model you defined.

</details>

<details>
<summary>Source 2: tutorial/query-param-models.md (Query Parameter Models { #query-parameter-models })</summary>

If you have a group of **query parameters** that are related, you can create a **Pydantic model** to declare them.

This would allow you to **re-use the model** in **multiple places** and also to declare validations and metadata for all the parameters at once. 😎

/// note

This is supported since FastAPI version `0.115.0`. 🤓

///

</details>

<details>
<summary>Source 3: release-notes.md (Release Notes > 0.115.0 (2024-09-17))</summary>

Now you can declare `Query`, `Header`, and `Cookie` parameters with Pydantic models. 🎉

Use Pydantic models for `Query` parameters:


```python
from typing import Annotated, Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

app = FastAPI()


class FilterParams(BaseModel):
    limit: int = Field(100, gt=0, le=100)
    offset: int = Field(0, ge=0)
    order_by: Literal["created_at", "updated_at"] = "created_at"
    tags: list[str] = []


@app.get("/items/")
async def read_items(filter_query: Annotated[FilterParams, Query()]):
    return filter_query
```
Read the new docs: [Query Parameter Models](https://fastapi.tiangolo.com/tutorial/query-param-models/).

Use Pydantic models for `Header` parameters:


```python
from typing import Annotated

from fastapi import FastAPI, Header
from pydantic import BaseModel

app = FastAPI()


class CommonHeaders(BaseModel):
    host: str
    save_data: bool
    if_modified_since: str | None = None
    traceparent: str | None = None
    x_tag: list[str] = []


@app.get("/items/")
async def read_items(headers: Annotated[CommonHeaders, Header()]):
    return headers
```
Read the new docs: [Header Parameter Models](https://fastapi.tiangolo.com/tutorial/header-param-models/).

Use Pydantic models for `Cookie` parameters:


```python
from typing import Annotated

from fastapi import Cookie, FastAPI
from pydantic import BaseModel

app = FastAPI()


class Cookies(BaseModel):
    session_id: str
    fatebook_tracker: str | None = None
    googall_tracker: str | None = None


@app.get("/items/")
async def read_items(cookies: Annotated[Cookies, Cookie()]):
    return cookies
```
Read the new docs: [Cookie Parameter Models](https://fastapi.tiangolo.com/tutorial/cookie-param-models/).

Use Pydantic models to restrict extra values for `Query` parameters (also applies to `Header` and `Cookie` parameters).

To achieve it, use Pydantic's `model_config = {"extra": "forbid"}`:


```python
from typing import Annotated, Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

app = FastAPI()


class FilterParams(BaseModel):
    model_config = {"extra": "forbid"}

    limit: int = Field(100, gt=0, le=100)
    offset: int = Field(0, ge=0)
    order_by: Literal["created_at", "updated_at"] = "created_at"
    tags: list[str] = []


@app.get("/items/")
async def read_items(filter_query: Annotated[FilterParams, Query()]):
    return filter_query
```
This applies to `Query`, `Header`, and `Cookie` parameters, read the new docs:

[Forbid Extra Query Parameters](https://fastapi.tiangolo.com/tutorial/query-param-models/#forbid-extra-query-parameters)

[Forbid Extra Headers](https://fastapi.tiangolo.com/tutorial/header-param-models/#forbid-extra-headers)

[Forbid Extra Cookies](https://fastapi.tiangolo.com/tutorial/cookie-param-models/#forbid-extra-cookies)

✨ Add support for Pydantic models for parameters using `Query`, `Cookie`, `Header`. PR [#12199](https://github.com/fastapi/fastapi/pull/12199) by [@tiangolo](https://github.com/tiangolo).

🌐 Add Portuguese translation for `docs/pt/docs/advanced/security/http-basic-auth.md`. PR [#12195](https://github.com/fastapi/fastapi/pull/12195) by [@ceb10n](https://github.com/ceb10n).

⬆ [pre-commit.ci] pre-commit autoupdate. PR [#12204](https://github.com/fastapi/fastapi/pull/12204) by [@pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci).

</details>

<details>
<summary>Source 4: tutorial/header-param-models.md (Header Parameter Models { #header-parameter-models } > Header Parameters with a Pydantic Model { #header-parameters-with-a-pydantic-model })</summary>

Declare the **header parameters** that you need in a **Pydantic model**, and then declare the parameter as `Header`:

{* ../../docs_src/header_param_models/tutorial001_an_py310.py hl[9:14,18] *}

**FastAPI** will **extract** the data for **each field** from the **headers** in the request and give you the Pydantic model you defined.

</details>

<details>
<summary>Source 5: tutorial/query-param-models.md (Query Parameter Models { #query-parameter-models } > Summary { #summary })</summary>

You can use **Pydantic models** to declare **query parameters** in **FastAPI**. 😎

/// tip

Spoiler alert: you can also use Pydantic models to declare cookies and headers, but you will read about that later in the tutorial. 🤫

///

</details>

**Answer:**

Here's how you can use Pydantic models for query parameters in FastAPI, based on the provided documentation:

1.  **Define a Pydantic Model:** Create a Pydantic `BaseModel` to represent the structure of your query parameters.  Declare the fields you expect in the model.

2.  **Annotate with `Query` (or `Header` or `Cookie`):**  Use the `Query` (or `Header` or `Cookie`) annotation on the Pydantic model. This tells FastAPI to extract the data from the query parameters (or headers or cookies) and map it to your model.

3.  **Use the Model in your Route:**  Pass the Pydantic model as a parameter to your route's function. FastAPI will automatically extract the data from the request and populate the model.

**Example (from Source 3):**

```python
from typing import Annotated, Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

app = FastAPI()


class FilterParams(BaseModel):
    limit: int = Field(100, gt=0, le=100)
    offset: int = Field(0, ge=0)
    order_by: Literal["created_at", "updated_at"] = "created_at"
    tags: list[str] = []


@app.get("/items/")
async def read_items(filter_query: Annotated[FilterParams, Query()]):
    return filter_query
```

In this example:

*   `FilterParams` is the Pydantic model.
*   `Query()` annotation tells FastAPI to treat `filter_query` as a query parameter.
*   The `Annotated[FilterParams, Query()]`  part indicates that the `filter_query` parameter should be of type `FilterParams` and should be extracted from the query parameters.

You can also use Pydantic models for `Header` and `Cookie` parameters following the same principles.

**Restricting Extra Parameters**

You can further control the query parameters by using `model_config = {"extra": "forbid"}` in your Pydantic model.  This will ensure that any extra query parameters that aren't defined in your model are rejected.  This is covered in Source 1 and demonstrated in the example code.

---
