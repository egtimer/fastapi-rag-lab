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


## j-11: "There was an error parsing the body" error on requesting endpoint in FastAPI

Question: [so_71063455](https://stackoverflow.com/questions/71063455/there-was-an-error-parsing-the-body-error-on-requesting-endpoint-in-fastapi) (CC BY-SA 4.0)

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
<summary>Source 2: release-notes.md (Release Notes > 0.123.2 (2025-12-02))</summary>

🐛 Fix unformatted `{type_}` in FastAPIError. PR [#14416](https://github.com/fastapi/fastapi/pull/14416) by [@Just-Helpful](https://github.com/Just-Helpful).

🐛 Fix parsing extra non-body parameter list. PR [#14356](https://github.com/fastapi/fastapi/pull/14356) by [@YuriiMotov](https://github.com/YuriiMotov).

🐛 Fix parsing extra `Form` parameter list. PR [#14303](https://github.com/fastapi/fastapi/pull/14303) by [@YuriiMotov](https://github.com/YuriiMotov).

🐛 Fix support for form values with empty strings interpreted as missing (`None` if that's the default), for compatibility with HTML forms. PR [#13537](https://github.com/fastapi/fastapi/pull/13537) by [@MarinPostma](https://github.com/MarinPostma).

📝 Add tip on how to install `pip` in case of `No module named pip` error in `virtual-environments.md`. PR [#14211](https://github.com/fastapi/fastapi/pull/14211) by [@zadevhub](https://github.com/zadevhub).

📝 Update Primary Key notes for the SQL databases tutorial to avoid confusion. PR [#14120](https://github.com/fastapi/fastapi/pull/14120) by [@FlaviusRaducu](https://github.com/FlaviusRaducu).

📝 Clarify estimation note in documentation. PR [#14070](https://github.com/fastapi/fastapi/pull/14070) by [@SaisakthiM](https://github.com/SaisakthiM).

</details>

<details>
<summary>Source 3: reference/exceptions.md (Exceptions - `HTTPException` and `WebSocketException`)</summary>

These are the exceptions that you can raise to show errors to the client.

When you raise an exception, as would happen with normal Python, the rest of the execution is aborted. This way you can raise these exceptions from anywhere in the code to abort a request and show the error to the client.

You can use:

`HTTPException`

`WebSocketException`

These exceptions can be imported directly from `fastapi`:


```python
from fastapi import HTTPException, WebSocketException
```
::: fastapi.HTTPException

::: fastapi.WebSocketException

</details>

<details>
<summary>Source 4: tutorial/body-multiple-params.md (Body - Multiple Parameters { #body-multiple-parameters } > Recap { #recap })</summary>

You can add multiple body parameters to your *path operation function*, even though a request can only have a single body.

But **FastAPI** will handle it, give you the correct data in your function, and validate and document the correct schema in the *path operation*.

You can also declare singular values to be received as part of the body.

And you can instruct **FastAPI** to embed the body in a key even when there is only a single parameter declared.

</details>

<details>
<summary>Source 5: deployment/concepts.md (Deployments Concepts { #deployments-concepts } > Restarts { #restarts })</summary>

Similar to making sure your application is run on startup, you probably also want to make sure it is **restarted** after failures.

We, as humans, make **mistakes**, all the time. Software almost *always* has **bugs** hidden in different places. 🐛

And we as developers keep improving the code as we find those bugs and as we implement new features (possibly adding new bugs too 😅).

When building web APIs with FastAPI, if there's an error in our code, FastAPI will normally contain it to the single request that triggered the error. 🛡

The client will get a **500 Internal Server Error** for that request, but the application will continue working for the next requests instead of just crashing completely.

Nevertheless, there might be cases where we write some code that **crashes the entire application** making Uvicorn and Python crash. 💥

And still, you would probably not want the application to stay dead because there was an error in one place, you probably want it to **continue running** at least for the *path operations* that are not broken.

But in those cases with really bad errors that crash the running **process**, you would want an external component that is in charge of **restarting** the process, at least a couple of times...

/// tip

...Although if the whole application is just **crashing immediately** it probably doesn't make sense to keep restarting it forever. But in those cases, you will probably notice it during development, or at least right after deployment.

So let's focus on the main cases, where it could crash entirely in some particular cases **in the future**, and it still makes sense to restart it.

///

You would probably want to have the thing in charge of restarting your application as an **external component**, because by that point, the same application with Uvicorn and Python already crashed, so there's nothing in the same code of the same app that could do anything about it.

In most cases, the same tool that is used to **run the program on startup** is also used to handle automatic **restarts**.

For example, this could be handled by:

Docker

Kubernetes

Docker Compose

Docker in Swarm Mode

Systemd

Supervisor

Handled internally by a cloud provider as part of their services

Others...

</details>

**Answer:**

According to the documentation, FastAPI internally raises a `RequestValidationError` when it encounters invalid data in a request. This `RequestValidationError` includes information about the file name and line where the error occurred, which can be helpful for logging and debugging. 

When you receive a "There was an error parsing the body" error, FastAPI provides a detailed JSON response that includes the location of the error (`loc`), an error message (`msg`), and the type of error (`type`).  You can also log the `body` of the invalid request to help with debugging. 

The `RequestValidationError` contains the `body` it received with invalid data. You could use it while developing your app to log the body and debug it, return it to the user, etc. 

Additionally, the documentation highlights that FastAPI's `HTTPException` error class inherits from Starlette's `HTTPException`, allowing you to handle Starlette exceptions within your FastAPI exception handlers.

---

## j-12: How to return a csv file/Pandas DataFrame in JSON format using FastAPI?

Question: [so_71203579](https://stackoverflow.com/questions/71203579/how-to-return-a-csv-file-pandas-dataframe-in-json-format-using-fastapi) (CC BY-SA 4.0)

<details>
<summary>Source 1: advanced/custom-response.md (Custom Response - HTML, Stream, File, others { #custom-response-html-stream-file-others })</summary>

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
<summary>Source 2: advanced/response-directly.md (Return a Response Directly { #return-a-response-directly })</summary>

When you create a **FastAPI** *path operation* you can normally return any data from it: a `dict`, a `list`, a Pydantic model, a database model, etc.

If you declare a [Response Model](../tutorial/response-model.md) FastAPI will use it to serialize the data to JSON, using Pydantic.

If you don't declare a response model, FastAPI will use the `jsonable_encoder` explained in [JSON Compatible Encoder](../tutorial/encoder.md) and put it in a `JSONResponse`.

You could also create a `JSONResponse` directly and return it.

/// tip

You will normally have much better performance using a [Response Model](../tutorial/response-model.md) than returning a `JSONResponse` directly, as that way it serializes the data using Pydantic, in Rust.

///

</details>

<details>
<summary>Source 3: advanced/response-directly.md (Return a Response Directly { #return-a-response-directly } > How a Response Model Works { #how-a-response-model-works })</summary>

When you declare a [Response Model - Return Type](../tutorial/response-model.md) in a path operation, **FastAPI** will use it to serialize the data to JSON, using Pydantic.

{* ../../docs_src/response_model/tutorial001_01_py310.py hl[16,21] *}

As that will happen on the Rust side, the performance will be much better than if it was done with regular Python and the `JSONResponse` class.

When using a `response_model` or return type, FastAPI won't use the `jsonable_encoder` to convert the data (which would be slower) nor the `JSONResponse` class.

Instead it takes the JSON bytes generated with Pydantic using the response model (or return type) and returns a `Response` with the right media type for JSON directly (`application/json`).

</details>

<details>
<summary>Source 4: advanced/custom-response.md (Custom Response - HTML, Stream, File, others { #custom-response-html-stream-file-others } > JSON Responses { #json-responses })</summary>

By default FastAPI returns JSON responses.

If you declare a [Response Model](../tutorial/response-model.md) FastAPI will use it to serialize the data to JSON, using Pydantic.

If you don't declare a response model, FastAPI will use the `jsonable_encoder` explained in [JSON Compatible Encoder](../tutorial/encoder.md) and put it in a `JSONResponse`.

If you declare a `response_class` with a JSON media type (`application/json`), like is the case with the `JSONResponse`, the data you return will be automatically converted (and filtered) with any Pydantic `response_model` that you declared in the *path operation decorator*. But the data won't be serialized to JSON bytes with Pydantic, instead it will be converted with the `jsonable_encoder` and then passed to the `JSONResponse` class, which will serialize it to bytes using the standard JSON library in Python.

In short, if you want the maximum performance, use a [Response Model](../tutorial/response-model.md) and don't declare a `response_class` in the *path operation decorator*.

{* ../../docs_src/response_model/tutorial001_01_py310.py ln[15:17] hl[16] *}

</details>

<details>
<summary>Source 5: advanced/dataclasses.md (Using Dataclasses { #using-dataclasses } > Dataclasses in Nested Data Structures { #dataclasses-in-nested-data-structures })</summary>

You can also combine `dataclasses` with other type annotations to make nested data structures.

In some cases, you might still have to use Pydantic's version of `dataclasses`. For example, if you have errors with the automatically generated API documentation.

In that case, you can simply swap the standard `dataclasses` with `pydantic.dataclasses`, which is a drop-in replacement:

{* ../../docs_src/dataclasses_/tutorial003_py310.py hl[1,4,7:10,13:16,22:24,27] *}

We still import `field` from standard `dataclasses`.

`pydantic.dataclasses` is a drop-in replacement for `dataclasses`.

The `Author` dataclass includes a list of `Item` dataclasses.

The `Author` dataclass is used as the `response_model` parameter.

You can use other standard type annotations with dataclasses as the request body.

In this case, it's a list of `Item` dataclasses.

Here we are returning a dictionary that contains `items` which is a list of dataclasses.

FastAPI is still capable of <dfn title="converting the data to a format that can be transmitted">serializing</dfn> the data to JSON.

Here the `response_model` is using a type annotation of a list of `Author` dataclasses.

Again, you can combine `dataclasses` with standard type annotations.

Notice that this *path operation function* uses regular `def` instead of `async def`.

As always, in FastAPI you can combine `def` and `async def` as needed.

If you need a refresher about when to use which, check out the section _"In a hurry?"_ in the docs about [`async` and `await`](../async.md#in-a-hurry).

This *path operation function* is not returning dataclasses (although it could), but a list of dictionaries with internal data.

FastAPI will use the `response_model` parameter (that includes dataclasses) to convert the response.

You can combine `dataclasses` with other type annotations in many different combinations to form complex data structures.

Check the in-code annotation tips above to see more specific details.

</details>

**Answer:**

Based on the provided context, here’s how to return a CSV file or Pandas DataFrame in JSON format using FastAPI:

1.  **Use a Response Model:** The most efficient approach is to declare a `Response Model` in your path operation. This allows FastAPI to serialize the data to JSON using Pydantic, which is much faster than using the `jsonable_encoder` or the `JSONResponse` class.  As Source 3 and 4 explain, this avoids the slower conversion processes.

2.  **Specify the Response Model:**  When you define the `response_model` parameter in your path operation decorator, you should specify a Pydantic model that represents the structure of your CSV data (likely a Pandas DataFrame).

3.  **Handle the DataFrame:** You would typically convert your Pandas DataFrame to a JSON-serializable format (e.g., a list of dictionaries) and assign it to the model.

4.  **Return a Response:** FastAPI will then automatically generate a `Response` with the correct `Content-Type` header (`application/json`) because the `response_model` tells FastAPI how to serialize the data.

**Example (Illustrative - Requires DataFrame Conversion):**

```python
from fastapi import FastAPI, Response
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

class Item(BaseModel):
    name: str
    value: int

@app.get("/items/", response_model=Item)
async def read_item():
    data = [Item(name="foo", value=1), Item(name="bar", value=2)]
    df = pd.DataFrame(data)
    return df.to_dict(orient='records') # Convert to list of dictionaries
```

**Important Note:** The context emphasizes that using a `Response Model` provides the best performance.  Avoid directly returning a `JSONResponse` unless you have a specific reason to use its features.

The context doesn't directly cover how to return a CSV file as JSON.  It assumes that the data is represented in a DataFrame, and you'd use the `response_model` to serialize the DataFrame to JSON.

---

## j-13: How to disable Swagger UI documentation in FastAPI for production server?

Question: [so_73677919](https://stackoverflow.com/questions/73677919/how-to-disable-swagger-ui-documentation-in-fastapi-for-production-server) (CC BY-SA 4.0)

<details>
<summary>Source 1: tutorial/metadata.md (Metadata and Docs URLs { #metadata-and-docs-urls } > Docs URLs { #docs-urls })</summary>

You can configure the two documentation user interfaces included:

**Swagger UI**: served at `/docs`.

You can set its URL with the parameter `docs_url`.

You can disable it by setting `docs_url=None`.

**ReDoc**: served at `/redoc`.

You can set its URL with the parameter `redoc_url`.

You can disable it by setting `redoc_url=None`.

For example, to set Swagger UI to be served at `/documentation` and disable ReDoc:

{* ../../docs_src/metadata/tutorial003_py310.py hl[3] *}

</details>

<details>
<summary>Source 2: how-to/configure-swagger-ui.md (Configure Swagger UI { #configure-swagger-ui } > Disable Syntax Highlighting { #disable-syntax-highlighting })</summary>

For example, you could disable syntax highlighting in Swagger UI.

Without changing the settings, syntax highlighting is enabled by default:

<img src="/img/tutorial/extending-openapi/image02.png">
But you can disable it by setting `syntaxHighlight` to `False`:

{* ../../docs_src/configure_swagger_ui/tutorial001_py310.py hl[3] *}

...and then Swagger UI won't show the syntax highlighting anymore:

<img src="/img/tutorial/extending-openapi/image03.png">

</details>

<details>
<summary>Source 3: tutorial/first-steps.md (First Steps { #first-steps } > Check it { #check-it } > Interactive API docs { #interactive-api-docs })</summary>

Now go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

You will see the automatic interactive API documentation (provided by [Swagger UI](https://github.com/swagger-api/swagger-ui)):

![Swagger UI](https://fastapi.tiangolo.com/img/index/index-01-swagger-ui-simple.png)

</details>

<details>
<summary>Source 4: release-notes.md (Release Notes > 0.104.1 (2023-10-30))</summary>

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
<summary>Source 5: reference/openapi/docs.md (OpenAPI `docs`)</summary>

Utilities to handle OpenAPI automatic UI documentation, including Swagger UI (by default at `/docs`) and ReDoc (by default at `/redoc`).

::: fastapi.openapi.docs.get_swagger_ui_html

::: fastapi.openapi.docs.get_redoc_html

::: fastapi.openapi.docs.get_swagger_ui_oauth2_redirect_html

::: fastapi.openapi.docs.swagger_ui_default_parameters

</details>

**Answer:**

You can disable Swagger UI by setting `docs_url=None`. 

Here's how it's documented:

"You can disable it by setting `docs_url=None`."

You can also configure the two documentation user interfaces included:

**Swagger UI**: served at `/docs`.

You can set its URL with the parameter `docs_url`.

---

## j-14: How to upload both file and JSON data using FastAPI?

Question: [so_70535711](https://stackoverflow.com/questions/70535711/how-to-upload-both-file-and-json-data-using-fastapi) (CC BY-SA 4.0)

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
<summary>Source 3: advanced/custom-response.md (Custom Response - HTML, Stream, File, others { #custom-response-html-stream-file-others } > JSON Responses { #json-responses })</summary>

By default FastAPI returns JSON responses.

If you declare a [Response Model](../tutorial/response-model.md) FastAPI will use it to serialize the data to JSON, using Pydantic.

If you don't declare a response model, FastAPI will use the `jsonable_encoder` explained in [JSON Compatible Encoder](../tutorial/encoder.md) and put it in a `JSONResponse`.

If you declare a `response_class` with a JSON media type (`application/json`), like is the case with the `JSONResponse`, the data you return will be automatically converted (and filtered) with any Pydantic `response_model` that you declared in the *path operation decorator*. But the data won't be serialized to JSON bytes with Pydantic, instead it will be converted with the `jsonable_encoder` and then passed to the `JSONResponse` class, which will serialize it to bytes using the standard JSON library in Python.

In short, if you want the maximum performance, use a [Response Model](../tutorial/response-model.md) and don't declare a `response_class` in the *path operation decorator*.

{* ../../docs_src/response_model/tutorial001_01_py310.py ln[15:17] hl[16] *}

</details>

<details>
<summary>Source 4: tutorial/request-files.md (Request Files { #request-files } > What is "Form Data" { #what-is-form-data })</summary>

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
<summary>Source 5: advanced/response-directly.md (Return a Response Directly { #return-a-response-directly })</summary>

When you create a **FastAPI** *path operation* you can normally return any data from it: a `dict`, a `list`, a Pydantic model, a database model, etc.

If you declare a [Response Model](../tutorial/response-model.md) FastAPI will use it to serialize the data to JSON, using Pydantic.

If you don't declare a response model, FastAPI will use the `jsonable_encoder` explained in [JSON Compatible Encoder](../tutorial/encoder.md) and put it in a `JSONResponse`.

You could also create a `JSONResponse` directly and return it.

/// tip

You will normally have much better performance using a [Response Model](../tutorial/response-model.md) than returning a `JSONResponse` directly, as that way it serializes the data using Pydantic, in Rust.

///

</details>

**Answer:**

To upload both files and JSON data using FastAPI, you can use the `File()` and `Form()` parameters together. FastAPI will handle the data using `multipart/form-data` encoding, which is the standard for sending files and form fields simultaneously. 

Here's how you would do it:

```python
from fastapi import FastAPI, File, Form

app = FastAPI()

@app.post("/items/")
async def create_item(
    name: str = Form(...),
    photo: UploadFile = File(...),
    audio: list[UploadFile] = Form(default=[]),
):
    return {"name": name, "photo": str(photo), "audio": audio}
```

**Explanation:**

*   `File(...)`:  This is used to handle individual file uploads.
*   `Form(...)`: This is used to handle form fields (JSON data).
*   `UploadFile`: This is a type hint for file uploads.  You can declare a single `UploadFile` or a list of `UploadFile`s for multiple file uploads.
*   `default=[]`: This is used to provide a default value for the `audio` parameter, ensuring that if no audio files are uploaded, it defaults to an empty list.

**Important Considerations:**

*   **multipart/form-data:** As highlighted in Sources 1 and 4, FastAPI will automatically encode the request body as `multipart/form-data` when files are included.  You don't need to manually construct this encoding.
*   **JSON and Files:** You can include both JSON data (using `Form()`) and file uploads (using `File()`) in the same request. FastAPI will separate these correctly.

Refer to Sources 1 and 2 for more detailed information on handling files and form data with FastAPI.

---

## j-15: FastAPI SQLAlchemy cannot convert dictionary update sequence element #0 to a sequence

Question: [so_76322342](https://stackoverflow.com/questions/76322342/fastapi-sqlalchemy-cannot-convert-dictionary-update-sequence-element-0-to-a-seq) (CC BY-SA 4.0)

<details>
<summary>Source 1: release-notes.md (Release Notes > 0.6.0 (2019-02-19))</summary>

Update SQL with SQLAlchemy tutorial at [https://fastapi.tiangolo.com/tutorial/sql-databases/](https://fastapi.tiangolo.com/tutorial/sql-databases/) using the new official `request.state`. PR [#45](https://github.com/tiangolo/fastapi/pull/45).

Upgrade Starlette to version `0.11.1` and add required compatibility changes. PR [#44](https://github.com/tiangolo/fastapi/pull/44).

</details>

<details>
<summary>Source 2: release-notes.md (Release Notes > 0.3.0 (2019-02-12))</summary>

Fix/add SQLAlchemy support, including ORM, and update [docs for SQLAlchemy](https://fastapi.tiangolo.com/tutorial/sql-databases/): [#30](https://github.com/tiangolo/fastapi/pull/30).

</details>

<details>
<summary>Source 3: release-notes.md (Release Notes > 0.123.3 (2025-12-02))</summary>

🐛 Fix Query\Header\Cookie parameter model alias. PR [#14360](https://github.com/fastapi/fastapi/pull/14360) by [@YuriiMotov](https://github.com/YuriiMotov).

🐛 Fix optional sequence handling in `serialize sequence value` with Pydantic V2. PR [#14297](https://github.com/fastapi/fastapi/pull/14297) by [@YuriiMotov](https://github.com/YuriiMotov).

</details>

<details>
<summary>Source 4: release-notes.md (Release Notes > 0.6.2 (2019-02-23))</summary>

Introduce new project generator based on FastAPI and PostgreSQL: [https://github.com/tiangolo/full-stack-fastapi-postgresql](https://github.com/tiangolo/full-stack-fastapi-postgresql). PR [#52](https://github.com/tiangolo/fastapi/pull/52).

Update [SQL tutorial with SQLAlchemy, using `Depends` to improve editor support and reduce code repetition](https://fastapi.tiangolo.com/tutorial/sql-databases/). PR [#52](https://github.com/tiangolo/fastapi/pull/52).

Improve middleware naming in tutorial for SQL with SQLAlchemy [https://fastapi.tiangolo.com/tutorial/sql-databases/](https://fastapi.tiangolo.com/tutorial/sql-databases/).

</details>

<details>
<summary>Source 5: release-notes.md (Release Notes > 0.30.0 (2019-06-20))</summary>

Add support for Pydantic's ORM mode:

Updated documentation about SQL with SQLAlchemy, using Pydantic models with ORM mode, SQLAlchemy models with relations, separation of files, simplification of code and other changes. New docs: [SQL (Relational) Databases](https://fastapi.tiangolo.com/tutorial/sql-databases/).

The new support for ORM mode fixes issues/adds features related to ORMs with lazy-loading, hybrid properties, dynamic/getters (using `@property` decorators) and several other use cases.

This applies to ORMs like SQLAlchemy, Peewee, Tortoise ORM, GINO ORM and virtually any other.

If your *path operations* return an arbitrary object with attributes (e.g. `my_item.name` instead of `my_item["name"]`) AND you use a `response_model`, make sure to update the Pydantic models with `orm_mode = True` as described in the docs (link above).

New documentation about receiving plain `dict`s as request bodies: [Bodies of arbitrary `dict`s](https://fastapi.tiangolo.com/tutorial/body-nested-models/#bodies-of-arbitrary-dicts).

New documentation about returning arbitrary `dict`s in responses: [Response with arbitrary `dict`](https://fastapi.tiangolo.com/tutorial/extra-models/#response-with-arbitrary-dict).

**Technical Details**:

When declaring a `response_model` it is used directly to generate the response content, from whatever was returned from the *path operation function*.

Before this, the return content was first passed through `jsonable_encoder` to ensure it was a "jsonable" object, like a `dict`, instead of an arbitrary object with attributes (like an ORM model). That's why you should make sure to update your Pydantic models for objects with attributes to use `orm_mode = True`.

If you don't have a `response_model`, the return object will still be passed through `jsonable_encoder` first.

When a `response_model` is declared, the same `response_model` type declaration won't be used as is, it will be "cloned" to create an new one (a cloned Pydantic `Field` with all the submodels cloned as well).

This avoids/fixes a potential security issue: as the returned object is passed directly to Pydantic, if the returned object was a subclass of the `response_model` (e.g. you return a `UserInDB` that inherits from `User` but contains extra fields, like `hashed_password`, and `User` is used in the `response_model`), it would still pass the validation (because `UserInDB` is a subclass of `User`) and the object would be returned as-is, including the `hashed_password`. To fix this, the declared `response_model` is cloned, if it is a Pydantic model class (or contains Pydantic model classes in it, e.g. in a `List[Item]`), the Pydantic model class(es) will be a different one (the "cloned" one). So, an object that is a subclass won't simply pass the validation and returned as-is, because it is no longer a sub-class of the cloned `response_model`. Instead, a new Pydantic model object will be created with the contents of the returned object. So, it will be a new object (made with the data from the returned one), and will be filtered by the cloned `response_model`, containing only the declared fields as normally.

PR [#322](https://github.com/tiangolo/fastapi/pull/322).

Remove/clean unused RegEx code in routing. PR [#314](https://github.com/tiangolo/fastapi/pull/314) by [@dmontagu](https://github.com/dmontagu).

Use default response status code descriptions for additional responses. PR [#313](https://github.com/tiangolo/fastapi/pull/313) by [@duxiaoyao](https://github.com/duxiaoyao).

Upgrade Pydantic support to `0.28`. PR [#320](https://github.com/tiangolo/fastapi/pull/320) by [@jekirl](https://github.com/jekirl).

</details>

**Answer:**

The context describes a fix for optional sequence handling in `serialize sequence value` with Pydantic V2. Specifically, PR [#14297](https://github.com/fastapi/fastapi/pull/14297) by [@YuriiMotov](https://github.com/YuriiMotov) addresses this issue.  To resolve this problem, you should update your Pydantic models with `orm_mode = True` if your *path operations* return an arbitrary object with attributes (e.g., `my_item.name` instead of `my_item["name"]`) and you use a `response_model`. This cloning mechanism prevents potential security issues and ensures proper validation.

---

## j-16: How to hide input param in FastApi?

Question: [so_66387641](https://stackoverflow.com/questions/66387641/how-to-hide-input-param-in-fastapi) (CC BY-SA 4.0)

<details>
<summary>Source 1: tutorial/response-model.md (Response Model - Return Type { #response-model-return-type } > Add an output model { #add-an-output-model })</summary>

We can instead create an input model with the plaintext password and an output model without it:

{* ../../docs_src/response_model/tutorial003_py310.py hl[9,11,16] *}

Here, even though our *path operation function* is returning the same input user that contains the password:

{* ../../docs_src/response_model/tutorial003_py310.py hl[24] *}

...we declared the `response_model` to be our model `UserOut`, that doesn't include the password:

{* ../../docs_src/response_model/tutorial003_py310.py hl[22] *}

So, **FastAPI** will take care of filtering out all the data that is not declared in the output model (using Pydantic).

In this case, because the two models are different, if we annotated the function return type as `UserOut`, the editor and tools would complain that we are returning an invalid type, as those are different classes.

That's why in this example we have to declare it in the `response_model` parameter.

...but continue reading below to see how to overcome that.

</details>

<details>
<summary>Source 2: release-notes.md (Release Notes > 0.103.0 (2023-08-26))</summary>

✨ Add support for `openapi_examples` in all FastAPI parameters. PR [#10152](https://github.com/tiangolo/fastapi/pull/10152) by [@tiangolo](https://github.com/tiangolo).

New docs: [OpenAPI-specific examples](https://fastapi.tiangolo.com/tutorial/schema-extra-example/#openapi-specific-examples).

📝 Add note to docs about Separate Input and Output Schemas with FastAPI version. PR [#10150](https://github.com/tiangolo/fastapi/pull/10150) by [@tiangolo](https://github.com/tiangolo).

</details>

<details>
<summary>Source 3: how-to/separate-openapi-schemas.md (Separate OpenAPI Schemas for Input and Output or Not { #separate-openapi-schemas-for-input-and-output-or-not } > Do not Separate Schemas { #do-not-separate-schemas })</summary>

Now, there are some cases where you might want to have the **same schema for input and output**.

Probably the main use case for this is if you already have some autogenerated client code/SDKs and you don't want to update all the autogenerated client code/SDKs yet, you probably will want to do it at some point, but maybe not right now.

In that case, you can disable this feature in **FastAPI**, with the parameter `separate_input_output_schemas=False`.

/// info

Support for `separate_input_output_schemas` was added in FastAPI `0.102.0`. 🤓

///

{* ../../docs_src/separate_openapi_schemas/tutorial002_py310.py hl[10] *}

And now there will be one single schema for input and output for the model, only `Item`, and it will have `description` as **not required**:

<div class="screenshot">
<img src="/img/tutorial/separate-openapi-schemas/image05.png">
</div>

</details>

<details>
<summary>Source 4: advanced/path-operation-advanced-configuration.md (Path Operation Advanced Configuration { #path-operation-advanced-configuration } > Advanced description from docstring { #advanced-description-from-docstring })</summary>

You can limit the lines used from the docstring of a *path operation function* for OpenAPI.

Adding an `\f` (an escaped "form feed" character) causes **FastAPI** to truncate the output used for OpenAPI at this point.

It won't show up in the documentation, but other tools (such as Sphinx) will be able to use the rest.

{* ../../docs_src/path_operation_advanced_configuration/tutorial004_py310.py hl[17:27] *}

</details>

<details>
<summary>Source 5: release-notes.md (Release Notes > 0.102.0 (2023-08-25))</summary>

✨ Add support for disabling the separation of input and output JSON Schemas in OpenAPI with Pydantic v2 with `separate_input_output_schemas=False`. PR [#10145](https://github.com/tiangolo/fastapi/pull/10145) by [@tiangolo](https://github.com/tiangolo).

New docs [Separate OpenAPI Schemas for Input and Output or Not](https://fastapi.tiangolo.com/how-to/separate-openapi-schemas/).

This PR also includes a new setup (internal tools) for generating screenshots for the docs.

♻️ Refactor tests for new Pydantic 2.2.1. PR [#10115](https://github.com/tiangolo/fastapi/pull/10115) by [@tiangolo](https://github.com/tiangolo).

📝 Add new docs section, How To - Recipes, move docs that don't have to be read by everyone to How To. PR [#10114](https://github.com/tiangolo/fastapi/pull/10114) by [@tiangolo](https://github.com/tiangolo).

📝 Update Advanced docs, add links to sponsor courses. PR [#10113](https://github.com/tiangolo/fastapi/pull/10113) by [@tiangolo](https://github.com/tiangolo).

📝 Update docs for generating clients. PR [#10112](https://github.com/tiangolo/fastapi/pull/10112) by [@tiangolo](https://github.com/tiangolo).

📝 Tweak MkDocs and add redirects. PR [#10111](https://github.com/tiangolo/fastapi/pull/10111) by [@tiangolo](https://github.com/tiangolo).

📝 Restructure docs for cloud providers, include links to sponsors. PR [#10110](https://github.com/tiangolo/fastapi/pull/10110) by [@tiangolo](https://github.com/tiangolo).

🔧 Update sponsors, add Speakeasy. PR [#10098](https://github.com/tiangolo/fastapi/pull/10098) by [@tiangolo](https://github.com/tiangolo).

</details>

**Answer:**

According to the documentation, you can hide input parameters in FastAPI by declaring a different output model than the input model. This is done using the `response_model` parameter in the path operation function. FastAPI will then filter out data not included in the output model using Pydantic.

Additionally, you can disable the separation of input and output schemas using the `separate_input_output_schemas=False` parameter, particularly useful when you have autogenerated client code/SDKs that you don't want to update immediately. This allows a single schema to be used for both input and output.

---

## j-17: How to implement OAuth to FastAPI with client ID & Secret

Question: [so_63248112](https://stackoverflow.com/questions/63248112/how-to-implement-oauth-to-fastapi-with-client-id-secret) (CC BY-SA 4.0)

<details>
<summary>Source 1: tutorial/security/first-steps.md (Security - First Steps { #security-first-steps } > **FastAPI**'s `OAuth2PasswordBearer` { #fastapis-oauth2passwordbearer })</summary>

**FastAPI** provides several tools, at different levels of abstraction, to implement these security features.

In this example we are going to use **OAuth2**, with the **Password** flow, using a **Bearer** token. We do that using the `OAuth2PasswordBearer` class.

/// info

A "bearer" token is not the only option.

But it's the best one for our use case.

And it might be the best for most use cases, unless you are an OAuth2 expert and know exactly why there's another option that better suits your needs.

In that case, **FastAPI** also provides you with the tools to build it.

///

When we create an instance of the `OAuth2PasswordBearer` class we pass in the `tokenUrl` parameter. This parameter contains the URL that the client (the frontend running in the user's browser) will use to send the `username` and `password` in order to get a token.

{* ../../docs_src/security/tutorial001_an_py310.py hl[8] *}

/// tip

Here `tokenUrl="token"` refers to a relative URL `token` that we haven't created yet. As it's a relative URL, it's equivalent to `./token`.

Because we are using a relative URL, if your API was located at `https://example.com/`, then it would refer to `https://example.com/token`. But if your API was located at `https://example.com/api/v1/`, then it would refer to `https://example.com/api/v1/token`.

Using a relative URL is important to make sure your application keeps working even in an advanced use case like [Behind a Proxy](../../advanced/behind-a-proxy.md).

///

This parameter doesn't create that endpoint / *path operation*, but declares that the URL `/token` will be the one that the client should use to get the token. That information is used in OpenAPI, and then in the interactive API documentation systems.

We will soon also create the actual path operation.

/// info

If you are a very strict "Pythonista" you might dislike the style of the parameter name `tokenUrl` instead of `token_url`.

That's because it is using the same name as in the OpenAPI spec. So that if you need to investigate more about any of these security schemes you can just copy and paste it to find more information about it.

///

The `oauth2_scheme` variable is an instance of `OAuth2PasswordBearer`, but it is also a "callable".

It could be called as:


```Python
oauth2_scheme(some, parameters)
```
So, it can be used with `Depends`.

Now you can pass that `oauth2_scheme` in a dependency with `Depends`.

{* ../../docs_src/security/tutorial001_an_py310.py hl[12] *}

This dependency will provide a `str` that is assigned to the parameter `token` of the *path operation function*.

**FastAPI** will know that it can use this dependency to define a "security scheme" in the OpenAPI schema (and the automatic API docs).

/// info | Technical Details

**FastAPI** will know that it can use the class `OAuth2PasswordBearer` (declared in a dependency) to define the security scheme in OpenAPI because it inherits from `fastapi.security.oauth2.OAuth2`, which in turn inherits from `fastapi.security.base.SecurityBase`.

All the security utilities that integrate with OpenAPI (and the automatic API docs) inherit from `SecurityBase`, that's how **FastAPI** can know how to integrate them in OpenAPI.

///

</details>

<details>
<summary>Source 2: tutorial/security/oauth2-jwt.md (OAuth2 with Password (and hashing), Bearer with JWT tokens { #oauth2-with-password-and-hashing-bearer-with-jwt-tokens } > Recap { #recap })</summary>

With what you have seen up to now, you can set up a secure **FastAPI** application using standards like OAuth2 and JWT.

In almost any framework handling the security becomes a rather complex subject quite quickly.

Many packages that simplify it a lot have to make many compromises with the data model, database, and available features. And some of these packages that simplify things too much actually have security flaws underneath.

**FastAPI** doesn't make any compromise with any database, data model or tool.

It gives you all the flexibility to choose the ones that fit your project the best.

And you can use directly many well maintained and widely used packages like `pwdlib` and `PyJWT`, because **FastAPI** doesn't require any complex mechanisms to integrate external packages.

But it provides you the tools to simplify the process as much as possible without compromising flexibility, robustness, or security.

And you can use and implement secure, standard protocols, like OAuth2 in a relatively simple way.

You can learn more in the **Advanced User Guide** about how to use OAuth2 "scopes", for a more fine-grained permission system, following these same standards. OAuth2 with scopes is the mechanism used by many big authentication providers, like Facebook, Google, GitHub, Microsoft, X (Twitter), etc. to authorize third party applications to interact with their APIs on behalf of their users.

</details>

<details>
<summary>Source 3: tutorial/security/first-steps.md (Security - First Steps { #security-first-steps })</summary>

Let's imagine that you have your **backend** API in some domain.

And you have a **frontend** in another domain or in a different path of the same domain (or in a mobile application).

And you want to have a way for the frontend to authenticate with the backend, using a **username** and **password**.

We can use **OAuth2** to build that with **FastAPI**.

But let's save you the time of reading the full long specification just to find those little pieces of information you need.

Let's use the tools provided by **FastAPI** to handle security.

</details>

<details>
<summary>Source 4: tutorial/sql-databases.md (SQL (Relational) Databases { #sql-relational-databases } > Update the App with Multiple Models { #update-the-app-with-multiple-models })</summary>

Now let's **refactor** this app a bit to increase **security** and **versatility**.

If you check the previous app, in the UI you can see that, up to now, it lets the client decide the `id` of the `Hero` to create. 😱

We shouldn't let that happen, they could overwrite an `id` we already have assigned in the DB. Deciding the `id` should be done by the **backend** or the **database**, **not by the client**.

Additionally, we create a `secret_name` for the hero, but so far, we are returning it everywhere, that's not very **secret**... 😅

We'll fix these things by adding a few **extra models**. Here's where SQLModel will shine. ✨

</details>

<details>
<summary>Source 5: reference/security/index.md (Security Tools > OpenID Connect)</summary>

::: fastapi.security.OpenIdConnect

</details>

**Answer:**

Based on the provided context, here's how you can implement OAuth2 with FastAPI using the Password flow and a Bearer token, including client ID and Secret:

1.  **Use `OAuth2PasswordBearer`:** The core of this implementation is the `OAuth2PasswordBearer` class. You instantiate it and pass in the `tokenUrl` parameter.

2.  **`tokenUrl` Parameter:** The `tokenUrl` parameter specifies the URL where the client (frontend) will send the `username` and `password` to obtain a token.  It's crucial to understand that this URL is relative. For example, if your FastAPI API is at `https://example.com/`, then the `tokenUrl` would be `token` (equivalent to `./token`).

3.  **Dependency with `Depends`:** The `oauth2_scheme` variable is an instance of `OAuth2PasswordBearer`, and it's a "callable" that can be used with `Depends`. This means you can pass it as a dependency to a path operation function. This dependency will provide a string assigned to the `token` parameter of the path operation.

4.  **OAuth2 Scopes (Advanced):** For a more fine-grained permission system, you can utilize OAuth2 with scopes, similar to what's used by providers like Facebook, Google, GitHub, etc. This allows you to define specific permissions that the frontend can request.

5.  **Security Best Practices:** The context emphasizes avoiding client-side control over IDs and secrets, and recommends backend/database-driven decisions. Also, you should avoid returning secrets everywhere.  Refactoring to ensure these principles is essential for robust security.

The provided context does not explicitly mention how to pass the client ID and Secret. However, OAuth2PasswordBearer typically relies on the username and password to get the token.  Consider implementing a backend authentication flow that would handle the OAuth2 flow, and pass the token to the frontend.

---

## j-18: List files from a static folder in FastAPI

Question: [so_71276790](https://stackoverflow.com/questions/71276790/list-files-from-a-static-folder-in-fastapi) (CC BY-SA 4.0)

<details>
<summary>Source 1: reference/staticfiles.md (Static Files - `StaticFiles`)</summary>

You can use the `StaticFiles` class to serve static files, like JavaScript, CSS, images, etc.

Read more about it in the [FastAPI docs for Static Files](https://fastapi.tiangolo.com/tutorial/static-files/).

You can import it directly from `fastapi.staticfiles`:


```python
from fastapi.staticfiles import StaticFiles
```
::: fastapi.staticfiles.StaticFiles

</details>

<details>
<summary>Source 2: tutorial/static-files.md (Static Files { #static-files } > Details { #details })</summary>

The first `"/static"` refers to the sub-path this "sub-application" will be "mounted" on. So, any path that starts with `"/static"` will be handled by it.

The `directory="static"` refers to the name of the directory that contains your static files.

The `name="static"` gives it a name that can be used internally by **FastAPI**.

All these parameters can be different than "`static`", adjust them with the needs and specific details of your own application.

</details>

<details>
<summary>Source 3: tutorial/static-files.md (Static Files { #static-files })</summary>

You can serve static files automatically from a directory using `StaticFiles`.

</details>

<details>
<summary>Source 4: tutorial/static-files.md (Static Files { #static-files } > More info { #more-info })</summary>

For more details and options check [Starlette's docs about Static Files](https://www.starlette.dev/staticfiles/).

</details>

<details>
<summary>Source 5: how-to/custom-docs-ui-assets.md (Custom Docs UI Static Assets (Self-Hosting) { #custom-docs-ui-static-assets-self-hosting } > Self-hosting JavaScript and CSS for docs { #self-hosting-javascript-and-css-for-docs })</summary>

Self-hosting the JavaScript and CSS could be useful if, for example, you need your app to keep working even while offline, without open Internet access, or in a local network.

Here you'll see how to serve those files yourself, in the same FastAPI app, and configure the docs to use them.

Let's say your project file structure looks like this:


```
.
├── app
│   ├── __init__.py
│   ├── main.py
```
Now create a directory to store those static files.

Your new file structure could look like this:


```
.
├── app
│   ├── __init__.py
│   ├── main.py
└── static/
```

Download the static files needed for the docs and put them on that `static/` directory.

You can probably right-click each link and select an option similar to "Save link as...".

**Swagger UI** uses the files:

[`swagger-ui-bundle.js`](https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js)

[`swagger-ui.css`](https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css)

And **ReDoc** uses the file:

[`redoc.standalone.js`](https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js)

After that, your file structure could look like:


```
.
├── app
│   ├── __init__.py
│   ├── main.py
└── static
    ├── redoc.standalone.js
    ├── swagger-ui-bundle.js
    └── swagger-ui.css
```

Import `StaticFiles`.

"Mount" a `StaticFiles()` instance in a specific path.

{* ../../docs_src/custom_docs_ui/tutorial002_py310.py hl[7,11] *}

Start your application and go to [http://127.0.0.1:8000/static/redoc.standalone.js](http://127.0.0.1:8000/static/redoc.standalone.js).

You should see a very long JavaScript file for **ReDoc**.

It could start with something like:


```JavaScript
/*! For license information please see redoc.standalone.js.LICENSE.txt */
!function(e,t){"object"==typeof exports&&"object"==typeof module?module.exports=t(require("null")):
...
```
That confirms that you are being able to serve static files from your app, and that you placed the static files for the docs in the correct place.

Now we can configure the app to use those static files for the docs.

The same as when using a custom CDN, the first step is to disable the automatic docs, as those use the CDN by default.

To disable them, set their URLs to `None` when creating your `FastAPI` app:

{* ../../docs_src/custom_docs_ui/tutorial002_py310.py hl[9] *}

And the same way as with a custom CDN, now you can create the *path operations* for the custom docs.

Again, you can reuse FastAPI's internal functions to create the HTML pages for the docs, and pass them the needed arguments:

`openapi_url`: the URL where the HTML page for the docs can get the OpenAPI schema for your API. You can use here the attribute `app.openapi_url`.

`title`: the title of your API.

`oauth2_redirect_url`: you can use `app.swagger_ui_oauth2_redirect_url` here to use the default.

`swagger_js_url`: the URL where the HTML for your Swagger UI docs can get the **JavaScript** file. **This is the one that your own app is now serving**.

`swagger_css_url`: the URL where the HTML for your Swagger UI docs can get the **CSS** file. **This is the one that your own app is now serving**.

And similarly for ReDoc...

{* ../../docs_src/custom_docs_ui/tutorial002_py310.py hl[2:6,14:22,25:27,30:36] *}

/// tip

The *path operation* for `swagger_ui_redirect` is a helper for when you use OAuth2.

If you integrate your API with an OAuth2 provider, you will be able to authenticate and come back to the API docs with the acquired credentials. And interact with it using the real OAuth2 authentication.

Swagger UI will handle it behind the scenes for you, but it needs this "redirect" helper.

///

Now, to be able to test that everything works, create a *path operation*:

{* ../../docs_src/custom_docs_ui/tutorial002_py310.py hl[39:41] *}

Now, you should be able to disconnect your WiFi, go to your docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), and reload the page.

And even without Internet, you would be able to see the docs for your API and interact with it.

</details>

**Answer:**

To serve static files from a directory using `StaticFiles` in FastAPI, you need to:

1.  Import `StaticFiles` from `fastapi.staticfiles`.
2.  Create a `StaticFiles()` instance, specifying the `directory` (the name of the directory where your static files are located) and the `"/static"` sub-path.  For example: `StaticFiles(directory="static", name="static")`.
3.  Mount the `StaticFiles()` instance in your FastAPI application, typically within your main application file (e.g., `main.py`).
4.  Access the static files using the `/static/<filename>` URL path.

For example, if your files are in a directory named "static" within your project, and you have a file named "redoc.standalone.js" in that directory, you would access it via the URL `http://127.0.0.1:8000/static/redoc.standalone.js`.

The context also provides an example of how the `StaticFiles` instance is used to serve files like `swagger-ui-bundle.js`, `swagger-ui.css`, and `redoc.standalone.js`.  It shows how to structure your project and how to configure the `StaticFiles` instance to serve these files.

---

## j-19: does fastapi contrib has paginator for postgresql?

Question: [so_60152442](https://stackoverflow.com/questions/60152442/does-fastapi-contrib-has-paginator-for-postgresql) (CC BY-SA 4.0)

<details>
<summary>Source 1: tutorial/sql-databases.md (SQL (Relational) Databases { #sql-relational-databases })</summary>

**FastAPI** doesn't require you to use a SQL (relational) database. But you can use **any database** that you want.

Here we'll see an example using [SQLModel](https://sqlmodel.tiangolo.com/).

**SQLModel** is built on top of [SQLAlchemy](https://www.sqlalchemy.org/) and Pydantic. It was made by the same author of **FastAPI** to be the perfect match for FastAPI applications that need to use **SQL databases**.

/// tip

You could use any other SQL or NoSQL database library you want (in some cases called <abbr title="Object Relational Mapper: a fancy term for a library where some classes represent SQL tables and instances represent rows in those tables">"ORMs"</abbr>), FastAPI doesn't force you to use anything. 😎

///

As SQLModel is based on SQLAlchemy, you can easily use **any database supported** by SQLAlchemy (which makes them also supported by SQLModel), like:

PostgreSQL

MySQL

SQLite

Oracle

Microsoft SQL Server, etc.

In this example, we'll use **SQLite**, because it uses a single file and Python has integrated support. So, you can copy this example and run it as is.

Later, for your production application, you might want to use a database server like **PostgreSQL**.

/// tip

There is an official project generator with **FastAPI** and **PostgreSQL** including a frontend and more tools: [https://github.com/fastapi/full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template)

///

This is a very simple and short tutorial, if you want to learn about databases in general, about SQL, or more advanced features, go to the [SQLModel docs](https://sqlmodel.tiangolo.com/).

</details>

<details>
<summary>Source 2: release-notes.md (Release Notes > 0.6.2 (2019-02-23))</summary>

Introduce new project generator based on FastAPI and PostgreSQL: [https://github.com/tiangolo/full-stack-fastapi-postgresql](https://github.com/tiangolo/full-stack-fastapi-postgresql). PR [#52](https://github.com/tiangolo/fastapi/pull/52).

Update [SQL tutorial with SQLAlchemy, using `Depends` to improve editor support and reduce code repetition](https://fastapi.tiangolo.com/tutorial/sql-databases/). PR [#52](https://github.com/tiangolo/fastapi/pull/52).

Improve middleware naming in tutorial for SQL with SQLAlchemy [https://fastapi.tiangolo.com/tutorial/sql-databases/](https://fastapi.tiangolo.com/tutorial/sql-databases/).

</details>

<details>
<summary>Source 3: tutorial/sql-databases.md (SQL (Relational) Databases { #sql-relational-databases })</summary>

**FastAPI** doesn't require you to use a SQL (relational) database. But you can use **any database** that you want.

Here we'll see an example using [SQLModel](https://sqlmodel.tiangolo.com/).

**SQLModel** is built on top of [SQLAlchemy](https://www.sqlalchemy.org/) and Pydantic. It was made by the same author of **FastAPI** to be the perfect match for FastAPI applications that need to use **SQL databases**.

/// tip

You could use any other SQL or NoSQL database library you want (in some cases called <abbr title="Object Relational Mapper: a fancy term for a library where some classes represent SQL tables and instances represent rows in those tables">"ORMs"</abbr>), FastAPI doesn't force you to use anything. 😎

///

As SQLModel is based on SQLAlchemy, you can easily use **any database supported** by SQLAlchemy (which makes them also supported by SQLModel), like:

PostgreSQL

MySQL

SQLite

Oracle

Microsoft SQL Server, etc.

In this example, we'll use **SQLite**, because it uses a single file and Python has integrated support. So, you can copy this example and run it as is.

Later, for your production application, you might want to use a database server like **PostgreSQL**.

/// tip

There is an official project generator with **FastAPI** and **PostgreSQL** including a frontend and more tools: [https://github.com/fastapi/full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template)

///

This is a very simple and short tutorial, if you want to learn about databases in general, about SQL, or more advanced features, go to the [SQLModel docs](https://sqlmodel.tiangolo.com/).

</details>

<details>
<summary>Source 4: project-generation.md (Full Stack FastAPI Template { #full-stack-fastapi-template } > Full Stack FastAPI Template - Technology Stack and Features { #full-stack-fastapi-template-technology-stack-and-features })</summary>

⚡ [**FastAPI**](https://fastapi.tiangolo.com) for the Python backend API.

🧰 [SQLModel](https://sqlmodel.tiangolo.com) for the Python SQL database interactions (ORM).

🔍 [Pydantic](https://docs.pydantic.dev), used by FastAPI, for the data validation and settings management.

💾 [PostgreSQL](https://www.postgresql.org) as the SQL database.

🚀 [React](https://react.dev) for the frontend.

💃 Using TypeScript, hooks, Vite, and other parts of a modern frontend stack.

🎨 [Tailwind CSS](https://tailwindcss.com) and [shadcn/ui](https://ui.shadcn.com) for the frontend components.

🤖 An automatically generated frontend client.

🧪 [Playwright](https://playwright.dev) for End-to-End testing.

🦇 Dark mode support.

🐋 [Docker Compose](https://www.docker.com) for development and production.

🔒 Secure password hashing by default.

🔑 JWT (JSON Web Token) authentication.

📫 Email based password recovery.

✅ Tests with [Pytest](https://pytest.org).

📞 [Traefik](https://traefik.io) as a reverse proxy / load balancer.

🚢 Deployment instructions using Docker Compose, including how to set up a frontend Traefik proxy to handle automatic HTTPS certificates.

🏭 CI (continuous integration) and CD (continuous deployment) based on GitHub Actions.

</details>

<details>
<summary>Source 5: how-to/graphql.md (GraphQL { #graphql })</summary>

As **FastAPI** is based on the **ASGI** standard, it's very easy to integrate any **GraphQL** library also compatible with ASGI.

You can combine normal FastAPI *path operations* with GraphQL on the same application.

/// tip

**GraphQL** solves some very specific use cases.

It has **advantages** and **disadvantages** when compared to common **web APIs**.

Make sure you evaluate if the **benefits** for your use case compensate the **drawbacks**. 🤓

///

</details>

**Answer:**

Yes, FastAPI doesn't require you to use a SQL database, but it can be used with any database that you want. It can be used with SQLModel which is built on top of SQLAlchemy and Pydantic. You can use databases like PostgreSQL, MySQL, SQLite, Oracle, and Microsoft SQL Server. 

The context also mentions a full-stack FastAPI and PostgreSQL project generator available at [https://github.com/fastapi/full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template) and another one at [https://github.com/tiangolo/full-stack-fastapi-postgresql](https://github.com/tiangolo/full-stack-fastapi-postgresql).

---

## j-20: fastapi.exceptions.FastAPIError: Invalid args for response field! Hint: check that <class 'main.SoluteSolvent'> is a valid pydantic field type

Question: [so_70609171](https://stackoverflow.com/questions/70609171/fastapi-exceptions-fastapierror-invalid-args-for-response-field-hint-check-th) (CC BY-SA 4.0)

<details>
<summary>Source 1: release-notes.md (Release Notes > 0.117.0 (2025-09-20))</summary>

✨  Allow `None` as return type for bodiless responses. PR [#9425](https://github.com/fastapi/fastapi/pull/9425) by [@hofrob](https://github.com/hofrob).

✨ Allow array values for OpenAPI schema `type` field. PR [#13639](https://github.com/fastapi/fastapi/pull/13639) by [@sammasak](https://github.com/sammasak).

✨ Add OpenAPI `external_docs` parameter to `FastAPI`. PR [#13713](https://github.com/fastapi/fastapi/pull/13713) by [@cmtoro](https://github.com/cmtoro).

⚡️ Fix `default_factory` for response model field with Pydantic V1. PR [#9704](https://github.com/fastapi/fastapi/pull/9704) by [@vvanglro](https://github.com/vvanglro).

🐛 Fix inconsistent processing of model docstring formfeed char with Pydantic V1. PR [#6039](https://github.com/fastapi/fastapi/pull/6039) by [@MaxwellPayne](https://github.com/MaxwellPayne).

🐛 Fix `jsonable_encoder` alters `json_encoders` of Pydantic v1 objects. PR [#4972](https://github.com/fastapi/fastapi/pull/4972) by [@aboubacs](https://github.com/aboubacs).

🐛 Reenable `allow_arbitrary_types` when only 1 argument is used on the API endpoint. PR [#13694](https://github.com/fastapi/fastapi/pull/13694) by [@rmawatson](https://github.com/rmawatson).

🐛 Fix `inspect.getcoroutinefunction()` can break testing with `unittest.mock.patch()`. PR [#14022](https://github.com/fastapi/fastapi/pull/14022) by [@secrett2633](https://github.com/secrett2633).

♻️ Create `dependency-cache` dict in `solve_dependencies` only if `None` (don't re-create if empty). PR [#13689](https://github.com/fastapi/fastapi/pull/13689) by [@bokshitsky](https://github.com/bokshitsky).

✅ Enable test case for duplicated headers in `test_tutorial/test_header_params/test_tutorial003.py`. PR [#13864](https://github.com/fastapi/fastapi/pull/13864) by [@Amogha-ark](https://github.com/Amogha-ark).

📌 Pin `httpx` to `>=0.23.0,<1.0.0`. PR [#14086](https://github.com/fastapi/fastapi/pull/14086) by [@YuriiMotov](https://github.com/YuriiMotov).

📝 Add note about Cookies and JavaScript on `tutorial/cookie-params.md`. PR [#13510](https://github.com/fastapi/fastapi/pull/13510) by [@Kludex](https://github.com/Kludex).

📝 Remove outdated formatting from `path-params-numeric-validations.md` for languages `en`, `es` and `uk`.. PR [#14059](https://github.com/fastapi/fastapi/pull/14059) by [@svlandeg](https://github.com/svlandeg).

📝 Fix and Improve English Documentation. PR [#14048](https://github.com/fastapi/fastapi/pull/14048) by [@nilslindemann](https://github.com/nilslindemann).

📝 Update prompts and German translation. PR [#14015](https://github.com/fastapi/fastapi/pull/14015) by [@nilslindemann](https://github.com/nilslindemann).

✅ Simplify tests for response_model. PR [#14062](https://github.com/fastapi/fastapi/pull/14062) by [@dynamicy](https://github.com/dynamicy).

🚨 Install pydantic.mypy plugin. PR [#14081](https://github.com/fastapi/fastapi/pull/14081) by [@svlandeg](https://github.com/svlandeg).

✅ Add LLM test file. PR [#14049](https://github.com/fastapi/fastapi/pull/14049) by [@nilslindemann](https://github.com/nilslindemann).

🔨 Update translations script. PR [#13968](https://github.com/fastapi/fastapi/pull/13968) by [@YuriiMotov](https://github.com/YuriiMotov).

🛠️ Update `docs.py generate-readme` command to remove permalinks from headers. PR [#14055](https://github.com/fastapi/fastapi/pull/14055) by [@YuriiMotov](https://github.com/YuriiMotov).

⬆️ Update mypy to 1.14.1. PR [#12970](https://github.com/fastapi/fastapi/pull/12970) by [@tamird](https://github.com/tamird).

</details>

<details>
<summary>Source 2: tutorial/response-model.md (Response Model - Return Type { #response-model-return-type } > Other Return Type Annotations { #other-return-type-annotations })</summary>

There might be cases where you return something that is not a valid Pydantic field and you annotate it in the function, only to get the support provided by tooling (the editor, mypy, etc).

The most common case would be [returning a Response directly as explained later in the advanced docs](../advanced/response-directly.md).

{* ../../docs_src/response_model/tutorial003_02_py310.py hl[8,10:11] *}

This simple case is handled automatically by FastAPI because the return type annotation is the class (or a subclass of) `Response`.

And tools will also be happy because both `RedirectResponse` and `JSONResponse` are subclasses of `Response`, so the type annotation is correct.

You can also use a subclass of `Response` in the type annotation:

{* ../../docs_src/response_model/tutorial003_03_py310.py hl[8:9] *}

This will also work because `RedirectResponse` is a subclass of `Response`, and FastAPI will automatically handle this simple case.

But when you return some other arbitrary object that is not a valid Pydantic type (e.g. a database object) and you annotate it like that in the function, FastAPI will try to create a Pydantic response model from that type annotation, and will fail.

The same would happen if you had something like a <dfn title='A union between multiple types means "any of these types".'>union</dfn> between different types where one or more of them are not valid Pydantic types, for example this would fail 💥:

{* ../../docs_src/response_model/tutorial003_04_py310.py hl[8] *}

...this fails because the type annotation is not a Pydantic type and is not just a single `Response` class or subclass, it's a union (any of the two) between a `Response` and a `dict`.

Continuing from the example above, you might not want to have the default data validation, documentation, filtering, etc. that is performed by FastAPI.

But you might want to still keep the return type annotation in the function to get the support from tools like editors and type checkers (e.g. mypy).

In this case, you can disable the response model generation by setting `response_model=None`:

{* ../../docs_src/response_model/tutorial003_05_py310.py hl[7] *}

This will make FastAPI skip the response model generation and that way you can have any return type annotations you need without it affecting your FastAPI application. 🤓

</details>

<details>
<summary>Source 3: tutorial/handling-errors.md (Handling Errors { #handling-errors } > Override the default exception handlers { #override-the-default-exception-handlers })</summary>

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
<summary>Source 4: tutorial/response-model.md (Response Model - Return Type { #response-model-return-type } > `response_model` Parameter { #response-model-parameter })</summary>

There are some cases where you need or want to return some data that is not exactly what the type declares.

For example, you could want to **return a dictionary** or a database object, but **declare it as a Pydantic model**. This way the Pydantic model would do all the data documentation, validation, etc. for the object that you returned (e.g. a dictionary or database object).

If you added the return type annotation, tools and editors would complain with a (correct) error telling you that your function is returning a type (e.g. a dict) that is different from what you declared (e.g. a Pydantic model).

In those cases, you can use the *path operation decorator* parameter `response_model` instead of the return type.

You can use the `response_model` parameter in any of the *path operations*:

`@app.get()`

`@app.post()`

`@app.put()`

`@app.delete()`

etc.

{* ../../docs_src/response_model/tutorial001_py310.py hl[17,22,24:27] *}

/// note

Notice that `response_model` is a parameter of the "decorator" method (`get`, `post`, etc). Not of your *path operation function*, like all the parameters and body.

///

`response_model` receives the same type you would declare for a Pydantic model field, so, it can be a Pydantic model, but it can also be, e.g. a `list` of Pydantic models, like `List[Item]`.

FastAPI will use this `response_model` to do all the data documentation, validation, etc. and also to **convert and filter the output data** to its type declaration.

/// tip

If you have strict type checks in your editor, mypy, etc, you can declare the function return type as `Any`.

That way you tell the editor that you are intentionally returning anything. But FastAPI will still do the data documentation, validation, filtering, etc. with the `response_model`.

///

If you declare both a return type and a `response_model`, the `response_model` will take priority and be used by FastAPI.

This way you can add correct type annotations to your functions even when you are returning a type different than the response model, to be used by the editor and tools like mypy. And still you can have FastAPI do the data validation, documentation, etc. using the `response_model`.

You can also use `response_model=None` to disable creating a response model for that *path operation*, you might need to do it if you are adding type annotations for things that are not valid Pydantic fields, you will see an example of that in one of the sections below.

</details>

<details>
<summary>Source 5: tutorial/response-model.md (Response Model - Return Type { #response-model-return-type } > Other Return Type Annotations { #other-return-type-annotations })</summary>

There might be cases where you return something that is not a valid Pydantic field and you annotate it in the function, only to get the support provided by tooling (the editor, mypy, etc).

The most common case would be [returning a Response directly as explained later in the advanced docs](../advanced/response-directly.md).

{* ../../docs_src/response_model/tutorial003_02_py310.py hl[8,10:11] *}

This simple case is handled automatically by FastAPI because the return type annotation is the class (or a subclass of) `Response`.

And tools will also be happy because both `RedirectResponse` and `JSONResponse` are subclasses of `Response`, so the type annotation is correct.

You can also use a subclass of `Response` in the type annotation:

{* ../../docs_src/response_model/tutorial003_03_py310.py hl[8:9] *}

This will also work because `RedirectResponse` is a subclass of `Response`, and FastAPI will automatically handle this simple case.

But when you return some other arbitrary object that is not a valid Pydantic type (e.g. a database object) and you annotate it like that in the function, FastAPI will try to create a Pydantic response model from that type annotation, and will fail.

The same would happen if you had something like a <dfn title='A union between multiple types means "any of these types".'>union</dfn> between different types where one or more of them are not valid Pydantic types, for example this would fail 💥:

{* ../../docs_src/response_model/tutorial003_04_py310.py hl[8] *}

...this fails because the type annotation is not a Pydantic type and is not just a single `Response` class or subclass, it's a union (any of the two) between a `Response` and a `dict`.

Continuing from the example above, you might not want to have the default data validation, documentation, filtering, etc. that is performed by FastAPI.

But you might want to still keep the return type annotation in the function to get the support from tools like editors and type checkers (e.g. mypy).

In this case, you can disable the response model generation by setting `response_model=None`:

{* ../../docs_src/response_model/tutorial003_05_py310.py hl[7] *}

This will make FastAPI skip the response model generation and that way you can have any return type annotations you need without it affecting your FastAPI application. 🤓

</details>

**Answer:**

The error "Invalid args for response field! Hint: check that <class 'main.SoluteSolvent'> is a valid Pydantic field type" indicates that FastAPI is trying to use a return type that isn't compatible with the `response_model` you've specified, or that isn't a valid Pydantic type.  Here's a breakdown of the likely causes and how to address them, based on the provided documentation:

**1. Mismatched Return Type and `response_model`:**

*   **The Problem:**  You've likely defined a return type annotation for your function that's not a standard Pydantic type (like `dict`, `list`, or a Pydantic model).  For example, you might be returning a custom class like `main.SoluteSolvent` (as the question states).  FastAPI expects the return type to be something it can readily convert and validate.
*   **The Solution:**  Use `response_model=None` to disable automatic response model generation. This allows you to return the `SoluteSolvent` object directly, without FastAPI trying to force a Pydantic model onto it.  This is often the fix when you are returning a custom object.

**2. Incorrect `response_model` Type:**

*   **The Problem:** You might be using an incorrect type for the `response_model` parameter.  It should be a Pydantic model class or a valid type like `list[Item]` (where `Item` is a Pydantic model).
*   **The Solution:** Ensure that the type specified in `response_model` is a valid Pydantic model or a type that FastAPI understands.

**3. Returning a Direct Response (Not a Pydantic Model):**

*   **The Problem:** You're trying to return a `Response` object directly (e.g., `RedirectResponse`, `JSONResponse`) instead of a Pydantic model or a data structure that FastAPI can convert to a response.
*   **The Solution:**  If you're returning a `Response` object, FastAPI handles the conversion automatically, provided the `Response` object has the expected fields.

**Applying the Solution (Based on the Question):**

Given the exact error message, the most likely solution is:

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response  # Import Response

# Assuming SoluteSolvent is defined in main.py
# ... your SoluteSolvent class definition ...

@app.get("/your_endpoint")
async def your_function(solute_solvent: main.SoluteSolvent):
    # Your logic here
    return Response(content=str(solute_solvent), media_type="text/plain") # Use Response to return plain text, not a Pydantic model
```

**Important Notes:**

*   **`Response`:** We're using `starlette.responses.Response` (as shown in the documentation) to create a raw `Response` object, and we specify `media_type="text/plain"` to return plain text.
*   **No `response_model`:** The key is to *remove* the `response_model` parameter from the function definition.

By following this approach, you allow FastAPI to handle the return of the `SoluteSolvent` object directly as the response, bypassing the validation and documentation issues that can arise when using an incompatible `response_model`.

---
