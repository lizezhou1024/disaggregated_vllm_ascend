# SPDX-License-Identifier: Apache-2.0

import os
import json

import aiohttp
from quart import Quart, make_response, request

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)

num_req = 0
num_req_prefilled = 0
num_req_fin = 0

prefill_port_base = 8100
decode_port_base = 8200

num_prefill_instance = int(os.getenv("NUM_PREFILL", "1"))
num_decode_instance = int(os.getenv("NUM_DECODE", "1"))

prefill_rank = 0
decode_rank = 0

async def forward_request(url, data, prev_data):
    skip_first_token = False
    record_first_token = False
    if prev_data:
        yield prev_data["resp"]
        skip_first_token = True
    else:
        record_first_token = True
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            #"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
            "User-Agent": "Benchmark Client"
        }
        async with session.post(url=url, json=data,
                                headers=headers) as response:
            if response.status == 200:
                # if response.headers.get('Transfer-Encoding') == 'chunked':
                if True:
                    async for chunk_bytes in response.content.iter_chunked(
                            1024):
                        if record_first_token:
                            prev_data["resp"] = chunk_bytes
                            record_first_token = False
                        if not skip_first_token:
                            yield chunk_bytes
                        else:
                            skip_first_token = False
                else:
                    content = await response.read()
                    yield content


@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request['max_tokens'] = 1

        global num_req
        global num_req_prefilled
        global num_req_fin
        num_req += 1
        my_req_id = num_req
        # print(f"Proxy Server: request count = {num_req}. ", flush=True)

        global prefill_rank
        global decode_rank

        my_prefill_rank = prefill_rank
        my_decode_rank = decode_rank

        prefill_rank = (prefill_rank + 1) % num_prefill_instance
        decode_rank = (decode_rank + 1) % num_decode_instance

        print(f"my_prefill_rank = {my_prefill_rank}, my_decode_rank = {my_decode_rank}")

        prefill_request['peer_kv_rank'] = my_decode_rank

        # finish prefill
        prefill_resp = {}
        async for _ in forward_request(f'http://localhost:{prefill_port_base+my_prefill_rank}/v1/completions',
                                       prefill_request,
                                       prefill_resp):
            continue

        num_req_prefilled += 1

        # print(f"Proxy Server: prefill request count = {num_req_prefilled}. ", flush=True)
        # print(f"Proxy Server: prefill request resp = {prefill_resp}. ", flush=True)
        # print(f"Proxy Server: start forward decoding {my_req_id}", flush=True)

        decode_request = original_request_data.copy()
        decode_request['peer_kv_rank'] = my_prefill_rank
        
        generator = forward_request(f'http://localhost:{decode_port_base+my_decode_rank}/v1/completions',
                                    decode_request, prefill_resp)
        response = await make_response(generator)
        response.timeout = None

        num_req_fin += 1
        # print(f"Proxy Server: fin request count = {num_req_prefilled}. ", flush=True)

        return response

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


if __name__ == '__main__':
    app.run(port=9000)
