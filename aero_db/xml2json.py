import xml.etree.ElementTree as ET
import json

def parse_xml_to_json(xml_file, output_json_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 데이터 구조 초기화
    aircraft_data = {}

    # 질량 및 관성 데이터
    mass_balance = root.find('mass_balance')
    aircraft_data["mass_balance"] = {
        "weight_lbs": float(mass_balance.find('emptywt').text),
        "cg_location": {
            "x_in": float(mass_balance.find('location/name[.="CG"]/x').text),
            "y_in": float(mass_balance.find('location/name[.="CG"]/y').text),
            "z_in": float(mass_balance.find('location/name[.="CG"]/z').text)
        },
        "inertia": {
            "ixx_slugft2": float(mass_balance.find('ixx').text),
            "iyy_slugft2": float(mass_balance.find('iyy').text),
            "izz_slugft2": float(mass_balance.find('izz').text),
            "ixz_slugft2": float(mass_balance.find('ixz').text)
        }
    }

    # 공력 데이터
    aerodynamics = root.find('aerodynamics')
    aero_data = {}
    for func in aerodynamics.findall('.//function'):
        func_name = func.get('name')
        table = func.find('table')
        if table is not None:
            independent_var = table.find('independentVar').text
            table_data = [list(map(float, row.text.split())) for row in table.findall('tableData')]
            aero_data[func_name.split('/')[-1]] = {
                "independentVar": independent_var,
                "data": table_data
            }
    aircraft_data["aerodynamics"] = aero_data

    # 추진 데이터
    propulsion = root.find('propulsion')
    engine = propulsion.find('engine')
    thrust_table = engine.find('.//table')
    aircraft_data["propulsion"] = {
        "engine": {
            "type": engine.get('type'),
            "thrust_table": {
                "independentVar": thrust_table.find('independentVar').text,
                "data": [list(map(float, row.text.split())) for row in thrust_table.findall('tableData')]
            },
            "fuel_flow_lbs_per_hr": float(engine.find('fuelflow').text) if engine.find('fuelflow') is not None else 0.0
        }
    }

    # 제어 시스템 데이터
    flight_control = root.find('flight_control')
    control_data = {}
    for channel in flight_control.findall('.//component'):
        comp_name = channel.get('name')
        table = channel.find('.//table')
        if table is not None:
            independent_var = table.find('independentVar').text
            table_data = [list(map(float, row.text.split())) for row in table.findall('tableData')]
            control_data[comp_name] = {
                "independentVar": independent_var,
                "data": table_data
            }
    aircraft_data["flight_control"] = control_data

    # JSON 파일로 저장
    with open(output_json_file, 'w') as json_file:
        json.dump({xml_file.split('.')[0]: aircraft_data}, json_file, indent=4)

# F-16과 F-15 파일 변환
parse_xml_to_json('f16.xml', 'f16.json')
parse_xml_to_json('f22.xml', 'f22.json')