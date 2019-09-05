import re, os, copy, xml.etree.ElementTree as ET
import Utility

# Configurations

properName = Utility.properName
shortHand = Utility.shortHand

source_name = 'main.carbon'
output_name = 'output.txt'

addCarbonCode = True
carbonCode = {shortHand: "cc", properName: "CarbonCode"}

defaultStack = Utility.defaultStack
defaultArgsStack = Utility.defaultArgsStack
defaultReturnStack = Utility.defaultReturnStack

inputGetter = "getInputContent()"
initPort = "__init__"

defaultAlgorithmTree = "algorithmTree"

class DirectiveTypes:
    src = "src"
    log = "log"

class Attributes:
    directive_type = "directive_type"
    location = "location"
    name = "name"
    handler = "handler"
    keys = "keys"

class CmdParser:
    reactiveCmdTypes = [Utility.Command.endIf, Utility.Command.elseIf, Utility.Command.ifNot] + [
        Utility.Command.endWhile] + [Utility.Command.endFunc, Utility.Command.func]

    allCmdTypes = reactiveCmdTypes + [Utility.Command.doIf, Utility.Command.doWhile, Utility.Command.returner,
                                      Utility.Command.cmd, Utility.Command.call, Utility.Command.input] + [Utility.Command.impasse]

    defaultCmdType = Utility.Command.cmd

    @staticmethod
    def validateCmdType(data, cmdType):
        for cmd in Utility.Command.types:
            if CmdParser.caseInsensitiveEqv(cmdType, cmd):
                return cmd

        raise Exception("Parser does not recognize cmd type \"{}\".".format(cmdType))

    @staticmethod
    def parseBounds_CarbonCode(carbonCode):
        stringToParse = "\n" + carbonCode + "\n"
        lineBreaks = [m.start() for m in re.finditer('\n', stringToParse)]

        cmdCount = 0
        lineBounds = {}

        for span in Utility.getAtLevel("[", "]", stringToParse, 1):
            blockStart, blockEnd = span
            cmdCount += 1

            hitBeginning = False
            hitEnd = False

            noOfBreaks = len(lineBreaks)

            lineStart = 0
            lineEnd = 0

            for i in range(len(lineBreaks)):
                if blockStart >= lineBreaks[i] and blockStart <= lineBreaks[i + 1] and not hitBeginning:
                    lineStart = i + 1
                    hitBeginning = True

                if blockEnd >= lineBreaks[i] and blockEnd <= lineBreaks[i + 1] and not hitEnd:
                    lineEnd = i + 1
                    hitEnd = True
                    break

            lineBound = Utility.LineBounds(lineStart, lineEnd)
            lineBounds[cmdCount] = lineBound
        return lineBounds

    @staticmethod
    def resolve_instructions(data, source):
        return Utility.InstructionParser.parse(data, source)

    @staticmethod
    def extract_from_unesc_parantheses(string):
        between_unescaped_parantheses = re.compile(r"(?<!\\)\((.*)(?<!\\)\)", re.DOTALL)

        match = between_unescaped_parantheses.search(string)
        if match is None:
            return ''

        return match.group(1)

    @staticmethod
    def resolve_evalutor(data, header):
        evaluatorBlock = CmdParser.extract_from_unesc_parantheses(header)
        return Utility.InstructionParser.resolveExpression(data, evaluatorBlock) + ";"

    @staticmethod
    def caseInsensitiveEqv(string_1, string_2):
        return string_1.lower() == string_2.lower()

    @staticmethod
    def splitHeaderAndContent(cmdBody):
        pattern = re.compile(r"(?<!\\):",
                             re.DOTALL)

        match = pattern.search(cmdBody)
        try:
            start, end = match.span()
            return cmdBody[:start], cmdBody[end:]

        except Exception as e:
            raise Exception("Body {} couldn't be split into header and content sub-sections.".format(cmdBody))

    @staticmethod
    def initializer_overhead_instructions():
        # Todo Use a seperator constant

        output = ""
        output += Utility.OpCodes.build + ": {}, {};\n".format(defaultStack[shortHand],
                                                              defaultStack[properName])
        output += Utility.OpCodes.build + ": {}, {};\n".format(defaultReturnStack[shortHand],
                                                              defaultReturnStack[properName])
        output += Utility.OpCodes.build + ": {}, {};\n".format(defaultArgsStack[shortHand],
                                                              defaultArgsStack[properName])
        return output

    @staticmethod
    def parseCmds(data, string, initOverhead):
        # Return None for port and cmd if they're not there but '' for the body
        pattern = re.compile(r"\[\s*(\*\*[a-zA-Z_][\w_]*\*\*)?\s*(\w+)?\s*(?<!\\):\s*(.*)\s*\]",
                             re.DOTALL)

        if addCarbonCode:
            ccLineBounds = CmdParser.parseBounds_CarbonCode(string)

        cmdMatches = []
        for spans in Utility.getAtLevel("[", "]", string, 1):
            start, end = spans
            cmd = string[start:end]
            cmdMatches.append(pattern.match(cmd))

        # Init Code
        initCode = CmdParser.initializer_overhead_instructions()
        initOverhead = Utility.InstructionParser.parse(data, initCode) + initOverhead

        output = ""
        openFunc = None # Used when building Return Cmds. Directs where to post returned value

        try:
            for match in cmdMatches:
                data.cmdCount += 1

                port = match.group(1)
                if port is not None:
                    # Remove leading and followings *s
                    port = port[2:-2]

                type = match.group(2)

                if type is None:
                    type = CmdParser.defaultCmdType

                # Type is case-corrected here
                type = CmdParser.validateCmdType(data, type)

                body = match.group(3)
                content = body

                cmdId = type.lower() + "_" + str(data.cmdCount)
                cmd = Utility.Command(cmdId, type)

                codeContent = {}

                codeContent[Utility.TemplateBuilders.execution_overhead] = ""
                codeContent[Utility.TemplateBuilders.cmd_id] = cmdId

                tail = ""

                # <------ Handling port
                if port is not None:
                    codeContent[Utility.TemplateBuilders.execution_overhead] += \
                        "clearOutput();\n"

                    if port == initPort:
                        codeContent[Utility.TemplateBuilders.execution_overhead] += initOverhead
                        tail += "{}.setInitializer({});".format(defaultAlgorithmTree, cmdId)
                    else:
                        tail += "{}.addAlgorithmHeader(\"{}\", {});".format(defaultAlgorithmTree, port, cmdId)

                    codeContent[Utility.TemplateBuilders.execution_overhead] += \
                        "\n" + Utility.VarParser.inflatePushVarsStack(defaultStack[properName])
                # ------>

                if type == Utility.Command.cmd:
                    pass

                elif type in [Utility.Command.doWhile, Utility.Command.doIf, Utility.Command.elseIf]:
                    header, content = CmdParser.splitHeaderAndContent(body)
                    codeContent[Utility.TemplateBuilders.evaluate] = CmdParser.resolve_evalutor(data,
                                                                                                header)

                elif type == Utility.Command.func:
                    header, content = CmdParser.splitHeaderAndContent(body)

                    parameterBlock = CmdParser.extract_from_unesc_parantheses(header)

                    try:
                        header = re.match(r"([\w ]+)(?<!\\)\(", header).group(1).strip()
                    except Exception as e:
                        raise Exception("Function header '{}' does not have parameters block defined.".format(
                            header))

                    returnType, funcIdentifier = header.split()

                    openFunc = funcIdentifier
                    Utility.VarParser.evalPushVarsStack(data, None)

                    if returnType not in Utility.Func.returnTypes:
                        raise Exception("Function return type '{}' is not recognized.".format(returnType))

                    postingReturn = ""
                    if returnType != Utility.Func.void:
                        returnVal = Utility.DataTypeMap.mapDefaults(returnType)
                        postingReturn = Utility.VarParser.inflateDimVar(funcIdentifier, returnType,
                                                                        defaultReturnStack[properName])

                    codeContent[Utility.TemplateBuilders.posting_return] = postingReturn

                    parameterNames = []
                    parameterTypes = []

                    if parameterBlock.strip() != "":
                        for parameter in parameterBlock.split(","):
                            parameter = parameter.strip()
                            parameterType, parameterName = parameter.split()

                            parameterNames.append(parameterName)
                            parameterTypes.append(parameterType)

                    localDataBuilding = Utility.VarParser.inflatePushVarsStack(
                        data.resolveStackKey(None)) + "\n"

                    dimCode = ""
                    for i in range(len(parameterNames)):
                        dimCode += Utility.OpCodes.dim + ": {} {}\n".format(parameterTypes[i],
                                                                            parameterNames[i])
                    localDataBuilding += Utility.InstructionParser.parse(data, dimCode)

                    setCode = ""
                    for i in range(len(parameterNames)):
                        getFromArgs = Utility.VarParser.inflateGetFromVars(parameterNames[i],
                                                                           parameterTypes[i],
                                                                           defaultArgsStack[properName])
                        setCode += Utility.OpCodes.set + ": {}, {}\n".format(parameterNames[i], getFromArgs)
                    localDataBuilding += Utility.InstructionParser.parse(data, setCode)

                    data.funcs[funcIdentifier] = Utility.Func(cmdId, funcIdentifier, parameterNames,
                                                              parameterTypes,
                                                              returnType)

                    tail += "{}.setIdentifier(\"{}\");".format(cmdId, funcIdentifier)
                    codeContent[Utility.TemplateBuilders.build_local_data] = localDataBuilding

                elif type == Utility.Command.call:
                    header, content = CmdParser.splitHeaderAndContent(body)

                    argumentsBlock = CmdParser.extract_from_unesc_parantheses(header)

                    try:
                        header = re.match(r"([\w ]+)(?<!\\)\(", header).group(1).strip()
                    except Exception as e:
                        raise Exception("Call header '{}' does not have arguments block defined.".format(
                            header))

                    funcIdentifier = header.strip()
                    data.validateFuncExists(funcIdentifier)

                    data.lastCalledFunc = funcIdentifier

                    func = data.funcs[funcIdentifier]

                    buildingArgs = Utility.VarParser.inflatePushVarsStack(
                        defaultArgsStack[properName]) + "\n"
                    passingArgs = ""

                    argCount = 0

                    if argumentsBlock.strip() != "":
                        for arg in argumentsBlock.split(","):
                            if argCount >= len(func.parameters):
                                raise Exception("Function '{}' expects {} arguments only.".format(
                                    funcIdentifier, len(func.parameters)))

                            arg = arg.strip()

                            parameterType = func.parameterTypes[argCount]
                            parameterName = func.parameters[argCount]

                            buildingArgs += Utility.VarParser.inflateDimVar(parameterName,
                                                                            parameterType,
                                                                            defaultArgsStack[properName])
                            buildingArgs += "\n"

                            # --- Append funcName to arg here and in func's retriever if you need it ---

                            getter = Utility.InstructionParser.resolveExpression(data, arg)
                            poster = Utility.VarParser.inflateSetVar(parameterName, parameterType,
                                                                     getter, defaultArgsStack[properName])
                            passingArgs += poster + "\n"
                            argCount += 1

                    tail = "\n{}.setFunction({});".format(cmdId, func.cmdId)
                    codeContent[Utility.TemplateBuilders.build_args] = buildingArgs
                    codeContent[Utility.TemplateBuilders.pass_args] = passingArgs

                elif type == Utility.Command.endFunc:
                    Utility.VarParser.evalPopVarsStack(data, None)

                    destroyingLocalData = Utility.VarParser.inflatePopVarsStack(
                        data.resolveStackKey(None)) + "\n"
                    destroyingLocalData += Utility.VarParser.inflatePopVarsStack(
                        defaultArgsStack[properName]) + "\n"

                    codeContent[Utility.TemplateBuilders.destroy_local_data] = destroyingLocalData

                elif type == Utility.Command.returner:
                    data.validateFuncExists(openFunc)
                    func = data.funcs[openFunc]

                    header, content = CmdParser.splitHeaderAndContent(body)
                    returnVal = Utility.InstructionParser.resolveExpression(
                        data, header.strip()).strip()

                    returnType = func.returnType

                    postingReturn = ""
                    if returnType != Utility.Func.void:
                        if returnVal == "":
                            returnVal = Utility.DataTypeMap.mapDefaults(returnType)

                        postingReturn = Utility.VarParser.inflateSetVar(func.identifier, returnType,
                                                                        returnVal,
                                                                        defaultReturnStack[properName])


                    codeContent[Utility.TemplateBuilders.posting_return] = postingReturn

                elif type == Utility.Command.input:
                    # Parsing input type and handling name of input is not being done ~ Here i am

                    header, content = CmdParser.splitHeaderAndContent(body)
                    inputType, inputIdentifier = header.strip().split()

                    if inputType not in Utility.DataTypeMap.getKeys():
                        raise Exception("Input type '{}' is not recognized.".format(
                            inputType))

                    processedInputType = Utility.DataTypeMap.mapDimType(inputType)

                    inputRetriever = inputGetter
                    if not Utility.addTypeBoxing:
                        inputRetriever = Utility.DataTypeMap.mapConverter(inputType) + "({})".format(
                            inputRetriever)

                    setDir = "{}, {}".format(inputIdentifier, "{}".format(inputRetriever))
                    postingInput = Utility.InstructionParser.evalSet(data, setDir)

                    codeContent[Utility.TemplateBuilders.input_type] = processedInputType
                    codeContent[Utility.TemplateBuilders.posting_input] = postingInput

                elif type == Utility.Command.impasse:
                    codeContent[Utility.TemplateBuilders.execution_overhead] += \
                    Utility.VarParser.inflatePopVarsStack(defaultStack[properName])

                    codeContent[Utility.TemplateBuilders.execution_overhead] += \
                        "\n" + Utility.LogParser.inflateLog("Algorithm Terminated")

                if addCarbonCode:
                    ccCmdBound = ccLineBounds[data.cmdCount]
                    start, end = ccCmdBound.start, ccCmdBound.end

                    codeContent[
                        Utility.TemplateBuilders.execution_overhead] += Utility.InstructionParser.evalHighlight(
                        data, "{} @ {} - {}".format(carbonCode[shortHand], start, end)
                    ) + "\n"

                template = Utility.Templates.getTemplate(type)
                codeContent[Utility.TemplateBuilders.execution] = CmdParser.resolve_instructions(data,
                                                                                                 content)

                output += template.format(**codeContent)

                if tail != "":
                    output += "\n" + tail + "\n"

                data.cmds.append(cmd)

            return output
        except Exception as e:
            raise Exception("Something went wrong when building command {}.\n".format(data.cmdCount) + str(e))

    @staticmethod
    def chainToNext(cmds, index):
        if (index + 1 < len(cmds) and cmds[index + 1].type not in CmdParser.reactiveCmdTypes):
            return "{}.chainTo({});".format(cmds[index].id, cmds[index + 1].id)

        return ""

    @staticmethod
    def functionSort(cmds, start, end):
        funcHead, funcTail = None, None

        for index in range(start, end):
            if cmds[index].type == Utility.Command.func:
                funcHead = index

            elif cmds[index].type == Utility.Command.endFunc:
                funcTail = index
                break

        if funcTail != None and funcHead != None:
            for index in range(funcHead, funcTail + 1):
                temp = cmds.pop(index)
                cmds.insert(start + index - funcHead, temp)

            CmdParser.functionSort(cmds, start + funcTail - funcHead + 1, end)

    @staticmethod
    def sortAllFunctions(cmds):
        newCmds = copy.deepcopy(cmds)
        CmdParser.functionSort(newCmds, 0, len(newCmds))

        return newCmds

    @staticmethod
    def chain(data):
        ifHeadsStack = []
        whileHeadsStack = []

        prevFuncHead = None

        output = ""

        funcSortedCmds = CmdParser.sortAllFunctions(data.cmds)

        for index in range(len(funcSortedCmds)):
            cmd = funcSortedCmds[index]
            type = cmd.type

            # Type is case-corrected here
            type = CmdParser.validateCmdType(data, type)

            prevIfHead = None
            if ifHeadsStack != []:
                prevIfHead = ifHeadsStack[len(ifHeadsStack) - 1]

            prevWhileHead = None
            if whileHeadsStack != []:
                prevWhileHead = whileHeadsStack[len(whileHeadsStack) - 1]

            if type == Utility.Command.doIf:
                ifHeadsStack.append(cmd)
                output += CmdParser.chainToNext(funcSortedCmds, index)

            elif type == Utility.Command.elseIf:
                if prevIfHead is None:
                    raise Exception("{} at cmd {} called without {}.".format(Utility.Command.elseIf, index,
                                                                             Utility.Command.doIf))

                output += ("{}.elseChainTo({});".format(prevIfHead.id, cmd.id))
                ifHeadsStack.append(cmd)

                output += CmdParser.chainToNext(funcSortedCmds, index)

            elif type == Utility.Command.ifNot:
                if prevIfHead is None:
                    raise Exception("{} at cmd {} called without {}.".format(Utility.Command.ifNot, index,
                                                                             Utility.Command.doIf))

                output += ("{}.elseChainTo({});".format(prevIfHead.id, cmd.id))

                output += CmdParser.chainToNext(funcSortedCmds, index)

            elif cmd.type == Utility.Command.endIf:
                if prevIfHead is None:
                    raise Exception("{} at cmd {} called without {}.".format(Utility.Command.elseIf, index,
                                                                             Utility.Command.endIf))

                cmdStackHeight = len(ifHeadsStack)

                for i in range(cmdStackHeight):
                    prevIfHead = ifHeadsStack[cmdStackHeight - i - 1]

                    if prevIfHead.type == Utility.Command.elseIf:
                        ifHeadsStack.pop()

                    elif prevIfHead.type == Utility.Command.doIf:
                        output += ("{}.endTo({});".format(prevIfHead.id, cmd.id))
                        break

                    else:
                        raise Exception("IfCmdsStack contain cmd of type {}.".format(cmd.type))

                if index + 1 < len(funcSortedCmds):
                    cmdNext = funcSortedCmds[index + 1]

                    if cmdNext.type not in CmdParser.reactiveCmdTypes:
                        output += ("{}.blockChainTo({});".format(prevIfHead.id, cmdNext.id))

                ifHeadsStack.pop()

            elif type == Utility.Command.doWhile:
                whileHeadsStack.append(cmd)
                output += CmdParser.chainToNext(funcSortedCmds, index)

            elif type == Utility.Command.endWhile:
                if prevWhileHead is None:
                    raise Exception("{} at cmd {} called without {}.".format(Utility.Command.endWhile, index,
                                                                             Utility.Command.doWhile))

                if index + 1 < len(funcSortedCmds):
                    cmdNext = funcSortedCmds[index + 1]

                    prevWhileHead = whileHeadsStack.pop()

                    if cmdNext.type not in CmdParser.reactiveCmdTypes:
                        output += ("{}.blockChainTo({});".format(prevWhileHead.id, cmdNext.id))

                output += ("{}.endTo({});".format(prevWhileHead.id, cmd.id))

            if type == Utility.Command.func:
                if prevFuncHead is not None or ifHeadsStack != [] or whileHeadsStack != []:
                    raise Exception("Trying to declare a function within a function/while/if block at cmd {" +
                                    "}.".format(index))

                prevFuncHead = cmd
                output += CmdParser.chainToNext(funcSortedCmds, index)

            elif type == Utility.Command.endFunc:
                if prevFuncHead is None:
                    raise Exception("{} at cmd {} called without {}.".format(Utility.Command.endFunc, index,
                                                                             Utility.Command.func))

                output += ("{}.endTo({});\n".format(prevFuncHead.id, cmd.id))

                prevFuncHead = None

            elif type == Utility.Command.returner:
                if prevFuncHead is None:
                    raise Exception("{} at cmd {} called without {}.".format(Utility.Command.returner, index,
                                                                             Utility.Command.func))

                # output += ("{}.returnAs({})".format(cmd.id, prevFuncHead.id))

            elif type == Utility.Command.call:
                output += CmdParser.chainToNext(funcSortedCmds, index)

            elif type == Utility.Command.input:
                output += CmdParser.chainToNext(funcSortedCmds, index)

            elif type == Utility.Command.cmd:
                output += CmdParser.chainToNext(funcSortedCmds, index)

            if type != Utility.Command.impasse:
                output += "\n"

        return output

    def srcDirective(data, attributes):
        shortHand = attributes[Attributes.handler]
        properName = attributes[Attributes.name]
        location = attributes[Attributes.location]

        srcFile = open(os.path.join(data.dir, location), "r")
        srcText = srcFile.read()
        srcFile.close()

        return Utility.CodeUnit(shortHand, properName, srcText)

    def logDirective(data, attributes):
        location = attributes[Attributes.location]

        srcFile = open(os.path.join(data.dir, location), "r")
        srcText = srcFile.read()
        srcFile.close()

        logTree = ET.fromstring(srcText)
        logs = logTree.findall('log')

        return logs

    def preProcess(data, code):
        directivePattern = re.compile(r"<~\s*([^~]*)?\s*~>", re.DOTALL)

        directives = []
        for directiveMatch in directivePattern.finditer(code):

            contentText = directiveMatch.group(1)
            attributes = Utility.resolveAttrFromContent(contentText)
            directives.append(attributes)

        return directives

    @staticmethod
    def parse(data, srcDir):
        output = ""

        main = open(os.path.join(srcDir, source_name), "r")
        text = main.read()
        main.close()

        data.dir = srcDir

        codeUnits = []
        directives = CmdParser.preProcess(data, text)

        for directive in directives:
            try:
                directiveType = directive[Attributes.directive_type]
                attributes = directive

                if CmdParser.caseInsensitiveEqv(directiveType, DirectiveTypes.src):
                    codeUnit = CmdParser.srcDirective(data, attributes)
                    codeUnits.append(codeUnit)

                if CmdParser.caseInsensitiveEqv(directiveType, DirectiveTypes.log):
                    data.logs += CmdParser.logDirective(data, attributes)

            except Exception as e:
                raise Exception("Something went wrong with handling directive \n\"{}\" \n".format(directive) +
                                str(e))

        if addCarbonCode:
            codeUnits.append(Utility.CodeUnit(carbonCode[shortHand],
                                              carbonCode[properName],
                                              text))

        initOverhead = r"// Building Code Units" + "\n" \
                       + Utility.CodeUnitParser.evalBuild(data, codeUnits) \
                       + "\n\n"

        output += CmdParser.parseCmds(data, text, initOverhead) + "\n\n"
        output += r"// Chaining" + "\n"
        output += CmdParser.chain(data)

        return output

dir = r"C:\Users\sakrit\OneDrive\Carbon 2.0\TestCC"

data = Utility.Data()

outputFile = open(os.path.join(dir, output_name), "w+")

outputText = CmdParser.parse(data, dir)
outputText = re.sub(r"\n\s*\n\s*\n", r"\n\n", outputText)
print(outputText)
outputFile.write(outputText)

outputFile.close()
