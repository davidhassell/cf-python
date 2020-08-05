import cfdm 


class Container: #cfdm.core.abstract.Container):
    '''Base class for storing components.

    .. versionadded:: 3.2.6

    '''
    def __docstring_substitution__(self):
        '''TODO

    Substitutions may be easily modified by overriding the
    __docstring_substitution__ method.

    Modifications can be applied to any class, and will only apply to
    that class and all of its subclases.

    If the key is a string then the special subtitutions will be
    applied to the dictionary values prior to replacement in the
    docstring.

    If the key is a compiled regular expession then the special
    subtitutions will be applied to the match of the regular
    expression prior to replacement in the docstring.

    For example:

       def __docstring_substitution__(self):
           def _upper(match):
               return match.group(1).upper()

           out = super().__docstring_substitution__()

           # Simple substitutions 
           out['{{repr}}'] = 'CF '
           out['{{foo}}'] = 'bar'

           out['{{parameter: `int`}}'] = """parameter: `int`
               This parameter does something to `{{class}}`
               instances. It has no default value.""",

           # Regular expression subsititions
           # 
           # Convert text to upper case
           out[re.compile('{{<upper (.*?)>}}')] = _upper

           return out

        '''
        return {
            '{{repr}}': 'CF ',
            
            '{{inplace: `bool`, optional}}':
            '''inplace: `bool`, optional
            If True then do the operation in-place and return `None`.''',

            '{{i: deprecated at version 3.0.0}}':
            '''i: deprecated at version 3.0.0
            Use the *inplace* parameter instead.''',
        }

# --- End: class
