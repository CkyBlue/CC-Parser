import re

# Keys
properName = "Proper Name"
shortHand = "Short Hand"

defaultStack = {shortHand: "l", properName: "Locals"}
defaultArgsStack = {shortHand: "a", properName: "args"}
defaultReturnStack = {shortHand: "r", properName: "return"}

# Config
key_for_fetching_return = "*RETURN"

class DataTypeMap:
    int = "INT"
    bool = "BOOL"
    str = "STR"
    float = "FLOAT"

    __typeDimMap = {int: "Type.INTEGER",
                    bool: "Type.BOOLEAN",
                    str: "Type.STRING",
                    float: "Type.FLOAT"}

    __getMap = {int: "getInt",
                bool: "getBool",
                str: "getStr",
                float: "getFloat"}

    __converterMap = {int: "Integer.valueOf",
                      bool: "Boolean.valueOf",
                      str: "String.valueOf",
                      float: "Float.valueOf"}

    @staticmethod
    def getKeys():
        return DataTypeMap.__typeDimMap.keys()

    @staticmethod
    def __validateKey(key):
        if key not in DataTypeMap.getKeys():
            raise KeyError(key + " is not a recognized key. The possible key values are:" +
                           str(DataTypeMap.getKeys()))

    @staticmethod
    def mapDimType(key):
        DataTypeMap.__validateKey(key)
        return DataTypeMap.__typeDimMap[key]

    @staticmethod
    def mapGetter(key):
        DataTypeMap.__validateKey(key)
        return DataTypeMap.__getMap[key]

    @staticmethod
    def mapConverter(key):
        DataTypeMap.__validateKey(key)
        return DataTypeMap.__converterMap[key]


class VarStack:
    def __init__(self, name):
        self.arrayOfVarMaps = [{}]
        self.name = name

    def push(self):
        self.arrayOfVarMaps.append({})

    def pop(self):
        self.arrayOfVarMaps.pop()

    def getMap(self):
        if len(self.arrayOfVarMaps) == 0:
            raise Exception("Stack is empty.")

        return self.arrayOfVarMaps[len(self.arrayOfVarMaps) - 1]


class Nodes:
    def __init__(self):
        self.size = 0
        self.name = ""
        self.components = {}


class Func:
    void = "VOID"

    returnTypes = [DataTypeMap.int,
                   DataTypeMap.str,
                   DataTypeMap.bool,
                   DataTypeMap.float,
                   void]

    def __init__(self, cmdId, identifier, parameters, parameterTypes, returnType):
        if len(parameters) != len(parameterTypes):
            raise Exception("No of parameters and parameter types do not match.")

        if returnType not in Func.returnTypes:
            raise Exception("Return type '{}' is not recognized.".format(returnType))

        for parameterType in parameterTypes:
            if parameterType not in DataTypeMap.getKeys():
                raise Exception("Parameter type '{}' is not recognized.".format(parameterType))

        self.cmdId = cmdId
        self.identifier = identifier
        self.parameters = parameters
        self.parameterTypes = parameterTypes
        self.returnType = returnType


class LineBounds:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class Command:
    call = "Call"
    cmd = "Cmd"
    endWhile = "EndWhile"
    endFunc = "EndFunc"
    endIf = "EndIf"
    doWhile = "While"
    func = "Func"
    doIf = "If"
    elseIf = "ElseIf"
    input = "Input"
    returner = "Return"
    ifNot = "Else"
    impasse = "Impasse"

    types = [call, cmd, endIf, endFunc, endWhile, doWhile, elseIf, func, doIf, input, returner, ifNot,
             impasse]

    def __init__(self, id, type):
        if type not in Command.types:
            raise Exception("Command of type '{}' is not recognized.".format(type))

        self.type = type
        self.id = id


class TemplateBuilders:
    __defaultClassModifiers = "final"
    __defaultMethodModifiers = "protected"
    void = "void"

    pass_args = "passArgs"
    build_args = "buildArgs"
    execution = "onExecution"
    evaluate = "evaluate"
    build_local_data = "buildLocalData"
    destroy_local_data = "destroyLocalData"
    input_type = "inputType"
    posting_return = "postingReturn"

    cmd_id = "cmdId"
    execution_overhead = "executionOverHead"

    __components = {pass_args: void + " " + pass_args + "() {{ {" + pass_args + "} }}",
                    build_args: void + " " + build_args + "() {{ {" + build_args + "} }}",
                    execution: void + " " + execution + "() {{ {" + execution_overhead + "} {" + execution + "} }}",
                    evaluate: "boolean " + evaluate + "() {{ return {" + evaluate + "} }}",
                    build_local_data: void + " " + build_local_data + "() {{ {" + build_local_data + "} }}",
                    posting_return: void + " " + execution + "() {{ {" + execution_overhead + "} {" + posting_return +
                                    "} {" + execution + "} }}",
                    destroy_local_data: void + " " + destroy_local_data + "() {{ {" + destroy_local_data + "} }}"}

    __classMap = {Command.call: "Call",
                  Command.cmd: "Command",
                  Command.endIf: "EndIf",
                  Command.endFunc: "EndFunction",
                  Command.endWhile: "EndWhile",
                  Command.doWhile: "DoWhile",
                  Command.elseIf: "IfOrElse",
                  Command.func: "Function",
                  Command.doIf: 'IfOrElse',
                  Command.input: "Input",
                  Command.returner: 'Command',
                  Command.ifNot: "Command",
                  Command.impasse: "Command"}

    @staticmethod
    def get_template_method(methodType):
        return "@Override\n" + TemplateBuilders.__defaultMethodModifiers + " " + \
               TemplateBuilders.__components[
                   methodType]

    @staticmethod
    def buildDefaultTemplate(type, components):
        className = TemplateBuilders.__classMap[type]
        template = TemplateBuilders.__defaultClassModifiers + " " + className + \
                   " {" + TemplateBuilders.cmd_id + "} = new " + className + '("{' + \
                   TemplateBuilders.cmd_id + '}") {{'

        for component in components:
            template += "\n" + TemplateBuilders.get_template_method(component) + "\n"

        template += "}};"
        return "\n" + template + "\n"

    @staticmethod
    def buildInputTemplate():
        inputCmdType = Command.input

        template = TemplateBuilders.__defaultClassModifiers + " " + inputCmdType + " {" + \
                   TemplateBuilders.cmd_id + "} = new " + \
                   inputCmdType + '("{' + TemplateBuilders.cmd_id + '}", {' + TemplateBuilders.input_type + '}) {{'

        template += "\n" + TemplateBuilders.get_template_method(TemplateBuilders.execution) + "\n"
        template += "}};"
        return "\n" + template + "\n"


class Templates:
    map = {
        Command.call: TemplateBuilders.buildDefaultTemplate(
            Command.call,
            [TemplateBuilders.execution,
             TemplateBuilders.build_args,
             TemplateBuilders.pass_args]),

        Command.endWhile: TemplateBuilders.buildDefaultTemplate(
            Command.endWhile,
            [TemplateBuilders.execution]),

        Command.cmd: TemplateBuilders.buildDefaultTemplate(
            Command.cmd,
            [TemplateBuilders.execution]),

        Command.endFunc: TemplateBuilders.buildDefaultTemplate(
            Command.endFunc,
            [TemplateBuilders.execution,
             TemplateBuilders.destroy_local_data]),

        Command.func: TemplateBuilders.buildDefaultTemplate(
            Command.func,
            [TemplateBuilders.execution,
             TemplateBuilders.build_local_data]),

        Command.doIf: TemplateBuilders.buildDefaultTemplate(
            Command.doIf,
            [TemplateBuilders.execution,
             TemplateBuilders.evaluate]),

        Command.ifNot: TemplateBuilders.buildDefaultTemplate(
            Command.ifNot,
            [TemplateBuilders.execution]),

        Command.doWhile: TemplateBuilders.buildDefaultTemplate(
            Command.doWhile,
            [TemplateBuilders.execution,
             TemplateBuilders.evaluate]),

        Command.elseIf: TemplateBuilders.buildDefaultTemplate(
            Command.elseIf,
            [TemplateBuilders.execution,
             TemplateBuilders.evaluate]),

        Command.endIf: TemplateBuilders.buildDefaultTemplate(
            Command.endIf,
            [TemplateBuilders.execution]),

        Command.input: TemplateBuilders.buildInputTemplate(),

        Command.returner: TemplateBuilders.buildDefaultTemplate(
            Command.returner,
            [TemplateBuilders.posting_return]),

        Command.impasse: TemplateBuilders.buildDefaultTemplate(
            Command.impasse,
            [])
    }

    @staticmethod
    def getTemplate(key):
        if key not in Templates.map.keys():
            raise Exception("Key {} not recognized.".format(key))

        return Templates.map[key]


class Data:
    def __init__(self):
        self.nodes = {}

        # Short-Hand -> Proper-Name
        self.varStackKeys = {}
        self.highLighterKeys = {}

        self.defaultVarStack = None
        # Is used for automatic pushing and popping in funciton calls.

        # Proper-Name -> VarStack
        self.varStacks = {}
        self.codeUnits = {}

        self.cmdCount = 0
        self.cmds = []

        self.funcs = {}
        self.lastCalledFunc = None

        self.dir = ""

    def validateVarExists(self, varName, stack):
        stack = Data.resolveStackKey(self, stack)
        if varName not in self.getVarMap(stack).keys():
            raise Exception("Variable with name \"{}\" in stack \"{}\"  is not recognized.".format(
                varName, stack))

    @staticmethod
    def validateIdentifier(name):
        pattern = re.compile(r'^([a-zA-Z_]\w*)$')
        if pattern.search(name) is None:
            raise Exception("Identifier '{}' is not valid.".format(name))

    def validateVarStackExists(self, stackName):
        stackName = self.resolveStackKey(stackName)

        if stackName not in self.varStacks.keys():
            raise Exception("VarStack of name \"{}\" doesn't exist.".format(stackName))

    def addVarStack(self, shortHand, properName):
        Data.validateIdentifier(properName)

        if shortHand in self.varStackKeys.keys() or properName in self.varStackKeys.values():
            raise Exception("Duplicate short hand '{}' or stack name '{}'".format(shortHand, properName))

        if self.defaultVarStack is None:
            self.defaultVarStack = shortHand

        self.varStackKeys[shortHand] = properName
        self.varStacks[properName] = VarStack(properName)

    def addVar(self, varName, stackKey, varType):
        stackName = self.resolveStackKey(stackKey)

        if varName in self.getVarMap(stackName).keys():
            raise Exception(
                "A variable with the name \"{}\" has already been declared in stack \"{}\".".format(
                    varName, stackName))

        if varType not in DataTypeMap.getKeys():
            raise Exception("Data type '{}' is not recognized.".format(varType))

        self.getVarMap(stackName)[varName] = varType

    def getVarMap(self, stackKey):
        stackName = self.resolveStackKey(stackKey)
        return self.varStacks[stackName].getMap()

    def getVarKeys(self, stackKey):
        stackName = self.resolveStackKey(stackKey)
        return self.getVarMap(stackName).keys()

    def getVarType(self, varName, stackKey):
        stackName = self.resolveStackKey(stackKey)
        self.validateVarExists(varName, stackName)

        return self.getVarMap(stackName)[varName]

    def pushVarStack(self, stackKey):
        stackName = self.resolveStackKey(stackKey)
        self.varStacks[stackName].push()

    def popVarStack(self, stackKey):
        stackName = self.resolveStackKey(stackKey)
        self.varStacks[stackName].pop()

    def resolveStackKey(self, stackKey):
        # Accepts either stack's Proper-Name or Short-Hand and returns Proper-Name

        if stackKey is None:
            if self.defaultVarStack is not None:
                return self.varStackKeys[self.defaultVarStack]

            else:
                raise Exception("Default VarStack has not been set yet.")

        stackKey = stackKey.strip()
        stackName = stackKey

        if stackKey not in self.varStackKeys.values():
            if stackKey not in self.varStackKeys.keys():
                raise Exception(" \"{}\" not a recognized stack name or short-hand.".format(stackKey))

            else:  # If key is short-hand
                stackName = self.varStackKeys[stackKey]

        return stackName

    def validateFuncExists(self, funcName):
        if funcName not in self.funcs.keys():
            raise Exception("Function '{}' is not recognized. Recognied function names are {}".format(
                funcName, self.funcs.keys()))

    def addCodeUnit(self, codeUnit):
        shortHand = codeUnit.shortHand
        properName = codeUnit.properName

        Data.validateIdentifier(properName)

        if shortHand in self.highLighterKeys.keys() or properName in self.highLighterKeys.values():
            raise Exception("Duplicate short hand '{}' or stack name '{}'".format(shortHand, properName))

        self.highLighterKeys[shortHand] = properName
        self.codeUnits[properName] = codeUnit

    def resolveCodeUnitKey(self, codeUnitKey):
        # Accepts either unit's Proper-Name or Short-Hand and returns Proper-Name

        codeUnitKey = codeUnitKey.strip()
        unitName = codeUnitKey

        if codeUnitKey not in self.highLighterKeys.values():
            if codeUnitKey not in self.highLighterKeys.keys():
                raise Exception(" \"{}\" not a recognized code unit name or short-hand.".format(codeUnitKey))

            else:  # If key is short-hand
                unitName = self.highLighterKeys[codeUnitKey]

        return unitName

    def getCodeUnit(self, codeUnitKey):
        # Accepts either unit's Proper-Name or Short-Hand and returns Proper-Name

        unitName = self.resolveCodeUnitKey(codeUnitKey)
        return self.codeUnits[unitName]


class VarParser:
    # Stack Key can be either a stack's Proper-Name or its Short-Hand

    @staticmethod
    def evalGetFromVars(data, address):
        varName, stackKey = VarParser.parseVarAddress(address)
        varType = data.getVarType(varName, stackKey)
        return VarParser.inflateGetFromVars(varName, varType, data.resolveStackKey(stackKey))

    @staticmethod
    def parseVarAddress(expression):
        try:
            if "@" not in expression:
                return expression.strip(), None

            varName, stackKey = expression.split("@")

            return varName.strip(), stackKey.strip()

        except Exception as e:
            raise Exception("Var address \"{}\" is invalid.".format(
                expression))

    @staticmethod
    def inflateGetFromVars(name, varType, stack):
        statement = VarParser.inflateGetVarsStack(stack) \
                    + ".{}(\"{}\")".format(DataTypeMap.mapGetter(varType), name)
        return statement

    @staticmethod
    def inflatePushVarsStack(stackName):
        return "pushVarStack(\"{}\");".format(stackName)

    @staticmethod
    def inflatePopVarsStack(stackName):
        return "popVarStack(\"{}\");".format(stackName)

    @staticmethod
    def evalBuildVarsStack(data, shortHand, properName):
        data.addVarStack(shortHand, properName)
        return "buildVarStack(\"{}\");".format(properName)

    @staticmethod
    def evalPushVarsStack(data, stackKey):
        data.pushVarStack(stackKey)
        return VarParser.inflatePushVarsStack(data.resolveStackKey(stackKey))

    @staticmethod
    def evalPopVarsStack(data, stackKey):
        data.popVarStack(stackKey)
        return VarParser.inflatePopVarsStack(data.resolveStackKey(stackKey))

    @staticmethod
    def evalDimVars(data, exp):
        varName, stackKey = VarParser.parseVarAddress(exp)
        varType, varName = varName.split()

        data.addVar(varName, stackKey, varType)
        return VarParser.inflateDimVar(varName, varType, data.resolveStackKey(stackKey))

    @staticmethod
    def inflateGetVarsStack(varStack):
        return "variables(\"{}\")".format(varStack)

    @staticmethod
    def inflateDimVar(varName, type, varStack):
        return VarParser.inflateGetVarsStack(varStack) + \
               ".declareVariable(\"{}\", {});".format(varName, DataTypeMap.mapDimType(type))

    @staticmethod
    def inflateSetVar(varName, varType, value, stack):
        typeCorrectedValue = DataTypeMap.mapConverter(varType) + "({})".format(value)
        return VarParser.inflateGetVarsStack(stack) + ".set(\"{}\", {});".format(varName, typeCorrectedValue)

    @staticmethod
    def evalSetVar(data, address, value):
        varName, stackKey = VarParser.parseVarAddress(address)
        varType = data.getVarType(varName, stackKey)
        return VarParser.inflateSetVar(varName, varType, value, data.resolveStackKey(stackKey))


def getAtLevel(startChar, endChar, s, requiredLevel):
    openCount = 0
    bound = {}

    if len(startChar) != 1 or len(endChar) != 1:
        raise Exception("The startChar and endChar arguments must be single characters.")

    for i in range(len(s)):
        c = s[i]
        if c == startChar:
            if i - 1 >= 0 and s[i - 1] == "\\":
                continue
            openCount += 1

            if openCount not in bound.keys():
                bound[openCount] = []

            bound[openCount].append([])
            bound[openCount][len(bound[openCount]) - 1].append(i)

        # print("Opening {} at {}".format(openCount, i))

        if c == endChar:
            if i - 1 >= 0 and s[i - 1] == "\\":
                continue
            # print("Closing {} at {}".format(openCount, i))

            if openCount not in bound.keys():
                raise Exception("Closing char at position {} doesn't have a matching opening.".format(i))

            bound[openCount][len(bound[openCount]) - 1].append(i + 1)

            openCount -= 1

    if requiredLevel not in bound.keys():
        return None

    # levels = max(bound.keys())

    # for level in range(levels, 0, -1):
    # 	print("Level: {} Bounds: {}".format(level, bound[level]))
    # 	for limits in bound[level]:
    # 		print("Level: {} Limits: {}".format(level, limits))

    return bound[requiredLevel]


class NodesParser:
    @staticmethod
    def validateNodesExists(data, nodesName):
        if nodesName not in data.nodes.keys():
            raise Exception("Nodes with name \"{}\" is not recognized.".format(nodesName))

    @staticmethod
    def evalGetFromNodes(data, exp):
        # Pre-Condition - Addresses within []s must be resolved of "[..]"s beforehand
        if "[" in exp:
            nodesName, col, index = NodesParser.parseNodeAddress(exp)
            NodesParser.validateNodesExists(data, nodesName)

            nodes = data.nodes[nodesName]
            colType = nodes.components[col]

            return NodesParser.inflateGetFromNode(nodesName, colType, col, index)

        else:
            nodesName = exp
            NodesParser.validateNodesExists(data, nodesName)

            return NodesParser.inflateGetNode(nodesName)

    @staticmethod
    def inflateGetFromNode(nodesName, colType, col, index):
        statement = NodesParser.inflateGetNode(nodesName) + ".{}(\"{}\", {})".format(
            DataTypeMap.mapGetter(colType), col, index)
        return statement

    @staticmethod
    def inflateGetNode(nodesName):
        return "nodes(\"{}\")".format(nodesName)

    @staticmethod
    def evalDimNodes(data, exp):
        pattern = re.compile(r'([a-zA-Z_]\w*)\s*@([\w\s|]+)@\s*(\d+)')

        match = re.search(pattern, exp)
        nodesName, columns, size = match.group(1), match.group(2), match.group(3)

        nodes = Nodes()
        nodes.size = int(size)
        nodes.name = nodesName

        data.nodes[nodesName] = nodes

        for col in columns.split("|"):
            col = col.strip()
            parts = col.split(" ")

            if len(parts) != 2:
                raise ValueError("Col declaration expression \"{}\" can only have 2 components.".format(exp))

            type = parts[0]
            name = parts[1]

            nodes.components[name] = type

        bluePrintName = "{}BluePrint".format(nodes.name.lower())

        buildBluePrint = "BluePrint " + bluePrintName + " = new BluePrint();\n"
        for colKey in nodes.components.keys():
            buildBluePrint += bluePrintName + '.addKey("{}", {});\n'.format(colKey,
                                                                            DataTypeMap.mapDimType(
                                                                                nodes.components[colKey]))

        buildNodes = buildBluePrint + "buildNodesStack(\"{}\", {}, {})".format(nodes.name, bluePrintName,
                                                                               nodes.size)

        return buildNodes + ";"

    @staticmethod
    def parseNodeAddress(expression):
        try:
            pattern = re.compile(r"([a-zA-Z_]\w*)\[([^<>]+)\]\[([^<>]+)\]")

            match = re.search(pattern, expression)
            nodesName, col, index = match.group(1), match.group(2), match.group(3)

            return nodesName, col, index

        except Exception as e:
            raise ValueError(expression + "is not a valid Node Address.")

    @staticmethod
    def evalSetToNode(data, nodeAddress, value):
        # Pre-Condition - Addresses within []s and value must be resolved of "[..]"s beforehand
        nodesName, col, index = NodesParser.parseNodeAddress(nodeAddress)
        NodesParser.validateNodesExists(data, nodesName)

        nodes = data.nodes[nodesName]
        colType = nodes.components[col]

        return NodesParser.inflateSetNodes(nodesName, col, colType, index, value)

    @staticmethod
    def inflateSetNodes(nodesName, col, colType, index, value):
        typeCorrectedValue = DataTypeMap.mapConverter(colType) + "({})".format(value)

        return NodesParser.inflateGetNode(nodesName) + ".set(\"{}\", {}, {});".format(col,
                                                                                      index,
                                                                                      typeCorrectedValue)


class CodeUnit:
    def __init__(self, shortHand, properName, text):
        self.shortHand = shortHand
        self.properName = properName
        self.text = text
        self.lineCount = len(text.split("\n"))


class CodeUnitParser:
    # Config
    tabSpaceEqv = 8

    @staticmethod
    def evalHighlight(data, expression):
        codeUnitKey, values = expression.split("@")

        codeUnitKey = data.resolveCodeUnitKey(codeUnitKey)

        values = values.strip()
        if values == "*":
            lineCount = data.getCodeUnit(codeUnitKey).lineCount
            values = "1"
            if lineCount > 1:
                values += " - {}".format(lineCount)

        if values == "!":
            statement = CodeUnitParser.inflateGetCodeUnit(codeUnitKey) + \
                        ".highlight(null)"

        else:
            lineNos = []
            valRanges = values.split(",")
            for varRange in valRanges:
                if "-" in varRange:
                    start, end = varRange.split("-")
                    start, end = int(start), int(end)

                    for i in range(start, end + 1):
                        lineNos.append(i)
                else:
                    lineNos.append(int(varRange))

            lineNos = set([str(lineNo) for lineNo in lineNos])
            lineNoStatement = ", ".join(lineNos)

            statement = CodeUnitParser.inflateGetCodeUnit(codeUnitKey) + \
                        ".highlight(new int[]{{ {} }});".format(lineNoStatement)

        return statement

    @staticmethod
    def evalBuild(data, codeUnits):
        unitNames = []

        for codeUnit in codeUnits:
            properName = codeUnit.properName
            unitNames.append(properName)

            data.addCodeUnit(codeUnit)

        output = "buildSourceCodeUnits(new String[]{{{}}});\n". \
            format(", ".join(['"' + unitName + '"' for unitName in unitNames]))

        for unitName in unitNames:
            srcText = data.getCodeUnit(unitName).text
            output += CodeUnitParser.evalSetCodeUnitText(data, srcText, unitName)

        return output + "\n"

    @staticmethod
    def inflateGetCodeUnit(codeUnit):
        return "getSourceCodeUnit(\"{}\")".format(codeUnit)

    @staticmethod
    def evalSetCodeUnitText(data, source, codeUnitKey):
        codeUnitName = data.resolveCodeUnitKey(codeUnitKey)
        codeUnitStrName = codeUnitName.lower() + "_str"

        source = re.sub(r"\\", r"\\\\", source)
        source = re.sub(r"\t", " " * CodeUnitParser.tabSpaceEqv, source)
        source = re.sub(r'"', r'\\"', source)

        lines = source.split("\n")

        output = "String " + codeUnitStrName + " = "

        for i in range(len(lines)):
            output += "\"{} \\n\" + \n"

        output = output.format(*lines) + '"";\n'
        output += CodeUnitParser.inflateGetCodeUnit(codeUnitName) \
                  + ".setText({});\n".format(codeUnitStrName)

        return output


class OpCodes:
    set = "SET"
    eval = "EVAL"
    dim = "DIM"
    build = "BUILD"
    light = "LIGHT"
    hide = "HIDE"
    show = "SHOW"
    log = "LOG"
    output = "OUTPUT"


class InstructionParser:
    opCodes = [OpCodes.set, OpCodes.eval,
               OpCodes.dim, OpCodes.build,
               OpCodes.light, OpCodes.hide,
               OpCodes.show, OpCodes.log,
               OpCodes.output]

    defaultOpCode = OpCodes.eval

    @staticmethod
    def getReturn(data, calledFunc):
        # --- Append funcName to arg here and in func's arg poster if you need it ---

        if calledFunc is None:
            raise Exception("No function was previously called.")

        func = data.funcs[calledFunc]
        type = func.returnType

        if type == Func.void:
            raise Exception("Attempting to access 'return' for a void function.")

        output = VarParser.inflateGetFromVars(calledFunc, type, defaultReturnStack[properName])
        return output


    @staticmethod
    def evalNodeAddress(data, exp):
        return NodesParser.evalGetFromNodes(data, exp)

    @staticmethod
    def evalVarAddress(data, exp):
        return VarParser.evalGetFromVars(data, exp)

    @staticmethod
    def parseAddress(data, exp):
        exp = exp.strip()

        if exp == key_for_fetching_return:
            return  InstructionParser.getReturn(data, )

        if len(exp) > 0 and exp[0] == "^":
            return InstructionParser.evalNodeAddress(data, exp[1:])

        else:
            return InstructionParser.evalVarAddress(data, exp)

    @staticmethod
    def resolveExpression(data, exp):
        opening_pattern = re.compile(r'(?<!\\)(<)')
        closing_pattern = re.compile(r'(?<!\\)(>)')

        while True:
            openings = []
            closings = []

            for match in opening_pattern.finditer(exp):
                openings.append(match.span(1)[0])

            for match in closing_pattern.finditer(exp):
                closings.append(match.span(1)[0])

            if len(openings) != len(closings):
                raise Exception("Number of <'s ({}) do not match the number of >'s ({}).".format(
                    len(openings),
                    len(closings)))

            elif len(openings) == 0:
                break

            innermost_end_bound = min(closings)
            innermost_start_bound = None

            for index in range(len(openings)):
                innermost_start_bound = openings[index]

                if index == len(openings) - 1 or (
                        openings[index] < innermost_end_bound and innermost_end_bound < openings[index + 1]):
                    break

            exp = exp[:innermost_start_bound] + \
                  InstructionParser. \
                      parseAddress(data, exp[innermost_start_bound + 1:innermost_end_bound]) + \
                  exp[innermost_end_bound + 1:]

        return exp

    @staticmethod
    def evalDim(data, exp):
        pattern = re.compile(r'(?<!\\)@')
        matches = pattern.findall(exp)

        if len(matches) == 2:
            return NodesParser.evalDimNodes(data, exp)

        return VarParser.evalDimVars(data, exp)

    @staticmethod
    def evalHighlight(data, exp):
        return CodeUnitParser.evalHighlight(data, exp)

    @staticmethod
    def evalLog(data, exp):
        return LogParser.evalLog(data, exp)

    @staticmethod
    def evalOutput(data, exp):
        return LogParser.evalOutput(data, exp)

    @staticmethod
    def evalBuild(data, exp):
        pattern = re.compile(r'\s*([a-zA-Z_]\w*)\s*(?<!\\),\s*([a-zA-Z_]\w*)\s*')
        match = pattern.match(exp)

        shortHand = match.group(1)
        actualName = match.group(2)

        return VarParser.evalBuildVarsStack(data, shortHand, actualName)

    @staticmethod
    def evalSet(data, exp):
        pattern = re.compile(r'\s*(.+)\s*(?<!\\),\s*(.+)\s*')
        match = pattern.match(exp)

        destinationAddrs = InstructionParser.resolveExpression(data, match.group(1))
        value = InstructionParser.resolveExpression(data, match.group(2))

        if "[" in destinationAddrs:
            setStatement = NodesParser.evalSetToNode(data, destinationAddrs, value)

        else:
            setStatement = VarParser.evalSetVar(data, destinationAddrs, value)

        return setStatement

    @staticmethod
    def evaluateExpression(data, exp):
        return InstructionParser.resolveExpression(data, exp) + ";"

    @staticmethod
    def parseInstruction(data, instruction):
        instruction = instruction.strip()

        pattern = re.compile(r'\s*(.*)\s*(?<!\\):\s*(.+)\s*')
        match = pattern.match(instruction)

        try:
            opCode = match.group(1)
            operand = match.group(2)

            if opCode == "" or opCode.isspace():
                opCode = InstructionParser.defaultOpCode

            if opCode not in InstructionParser.opCodes:
                raise Exception("OpCode '{}' not recognized.".format(opCode))

            directives = {OpCodes.set: InstructionParser.evalSet,
                          OpCodes.eval: InstructionParser.evaluateExpression,
                          OpCodes.dim: InstructionParser.evalDim,
                          OpCodes.build: InstructionParser.evalBuild,
                          OpCodes.light: InstructionParser.evalHighlight,
                          OpCodes.log: InstructionParser.evalLog,
                          OpCodes.output: InstructionParser.evalOutput,
                          OpCodes.hide: None,
                          OpCodes.show: None}

            operator = directives[opCode]
            output = operator(data, operand)

            return re.sub(r"\\<", r"<", output)

        except Exception as e:
            raise Exception("Instruction '{}' at cmd {} could not be parsed: ".format(
                instruction, data.cmdCount) + str(e))

    @staticmethod
    def parse(data, instructions):
        output = ""

        for line in instructions.split("\n"):
            if line == "" or line.isspace():
                output += "\n"
                continue

            output += InstructionParser.parseInstruction(data, line) + "\n"

        return output

    @staticmethod
    def mapGetter(varType):
        return DataTypeMap.mapGetter(varType)

    @staticmethod
    def mapDimType(varType):
        return DataTypeMap.mapDimType(varType)


class LogParser:
    @staticmethod
    def eval(data, log):
        while True:
            spans = getAtLevel("<", ">", log, 1)
            if spans is None:
                break
            else:
                start, end = spans.pop()

                currentExp = log[start:end]
                evaluation = InstructionParser.evaluateExpression(data, currentExp)[:-1]

                log = log[:start] + '" + {} + "'.format(evaluation) + log[end:]

        return log

    @staticmethod
    def inflateLog(data, log):
        return "log(\"{}\");".format(log)

    @staticmethod
    def inflateOutput(data, log):
        return "output(\"{}\");".format(log)

    @staticmethod
    def evalLog(data, log):
        resolvedLog = LogParser.eval(data, log)
        return LogParser.inflateLog(data, resolvedLog)

    @staticmethod
    def evalOutput(data, log):
        resolvedLog = LogParser.eval(data, log)
        return LogParser.inflateOutput(data, resolvedLog)


if __name__ == "__main__":
    for cmdKey in Command.types:
        print("<" + "-" * 24)
        print(Templates.getTemplate(cmdKey))
        print("-" * 24 + ">")

    data = Data()

    output = "\n"

    output += VarParser.evalBuildVarsStack(data, "l", "Locals") + "\n"
    output += VarParser.evalBuildVarsStack(data, "a", "Args") + "\n"
    output += VarParser.evalDimVars(data, "INT x") + "\n"
    output += VarParser.evalSetVar(data, "x", "24") + "\n"
    output += VarParser.evalPushVarsStack(data, None) + "\n"
    output += VarParser.evalDimVars(data, "INT x") + "\n"
    output += VarParser.evalSetVar(data, "x", "20") + "\n"
    output += VarParser.evalPopVarsStack(data, None) + "\n"
    output += VarParser.evalGetFromVars(data, "x") + "\n"

    output += '\n'

    output += NodesParser.evalDimNodes(data, "List @ INT x @ 12") + "\n"
    output += NodesParser.evalSetToNode(data, "List[x][11]", "99") + "\n"
    output += NodesParser.evalGetFromNodes(data, "List[x][11]") + "\n"
    output += NodesParser.evalGetFromNodes(data, "List") + "\n"

    output += '\n'

    codeUnits = [CodeUnit("p", "Python", "Something \" \" \n" * 4),
                 CodeUnit("cc", "CarbonCode", "\tPseudoCode \\ \n" * 4)]
    output += CodeUnitParser.evalBuild(data, codeUnits)
    output += CodeUnitParser.evalHighlight(data, "p @ 1 - 1")

    output += '\n'
    output += LogParser.evalLog(data, "Hello <^List[x][<x>]>")

    print(output)
