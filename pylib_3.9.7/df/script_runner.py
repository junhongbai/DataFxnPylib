import importlib
import sys

from df.data_transfer import DataFunctionRequest, DataFunction, open_json

if __name__ == '__main__':
    if len(sys.argv) == 3:
        in_file = sys.argv[1]
        out_file = sys.argv[2]
    else:
        in_file = 'in.json'
        out_file = 'out.json'

    with open_json(in_file, 'r') as fh:
        encoding = fh.encoding
        request_json = fh.read()

    request = DataFunctionRequest.parse_raw(request_json)  # type: DataFunctionRequest
    class_name = request.serviceName
    module = importlib.import_module(f'df.{class_name}')
    class_ = getattr(module, class_name)
    df: DataFunction = class_()
    response = df.execute(request)
    response_json = response.json()

    with open(out_file, 'w', encoding=encoding) as fh:
        fh.write(response_json)
