#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/05/17 at 11:04 AM

@author: neil

Program description here

Version 0.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.io.votable import exceptions as EE
from astropy import units as u
from tqdm import tqdm
import collections
import requests
import warnings
# warnings.filterwarnings("error")
from formlayout import fedit
import itertools


# =============================================================================
# Define variables
# =============================================================================
WORKSPACE = '/Astro/Projects/RayAleks_Work/UpperSco/'
DATAPATH = WORKSPACE + '/Data/Remastered_catalogue/From_lit_1/'
DATAFILE = 'LitMaster_with_bestpms_wise_isofit_disk_expm.votable'
SAVEPATH = DATAPATH
SAVEFILE = DATAFILE

# -----------------------------------------------------------------------------
# URL for UCD list
UCD_URL = 'http://cdsweb.u-strasbg.fr/UCD/ucd1p-words.txt'

FORML = 25

# =============================================================================
# Define functions
# =============================================================================
def deal_with_astropy_warnings(table, metadata, warning, valid_ucds):
    names, units, ucds = metadata
    # see if we can deal with exception
    if warning.category == EE.W03:
        return deal_with_w03(table, names, warning)
    elif warning.category == EE.W06:
        return deal_with_w06(table, ucds, warning, valid_ucds)
    elif warning.category == EE.W50:
        return deal_with_w50(table, units, warning)
    else:
        return table


def deal_with_w03(table, names, warning):
    """
    Deal with W03: Implicitly generating an ID from a name ‘x’ -> ‘y’
    
    :param table: the astropy table
    :param warning: the astropy.io.votable.exceptions.W03 warning
    
    ID and name are different --> do nothing
    
    Details here: 
    http://docs.astropy.org/en/stable/io/votable/api_exceptions.html#
    w03-implicitly-generating-an-id-from-a-name-x-y
    
    :return: 
    """
    return table


def deal_with_w06(table, ucds, warning, valid_ucds):
    """
    Deal with W06: Invalid UCD ‘x’: explanation
    
    :param table: the astropy table
    :param ucds: dictionary of ucd meta data (from votable meta data)
    :param warning: the astropy.io.votable.exceptions.W06 warning
    :params valid_ucds: dictionary, all valid ucds, keys required are
                        'ucd' the ucd value and 'desc' the descriptions
    
    ucd does not match syntax of unified content descriptor
    
    Details here:
    http://docs.astropy.org/en/stable/io/votable/api_exceptions.html#
    w06-invalid-ucd-x-explanation
    
    :return: 
    """

    # extract ucd
    raw = str(warning.message).split("W06: Invalid UCD '")[-1]
    strucd = raw.split("'")[0]
    strerror = raw.split("Unknown word '")[-1].split("'")[0]
    strcomponents = strerror.split('.')
    strother = strucd.replace(strerror, '{0}')

    # find columns containing strucd
    cols = find_ucd_in_dict(strucd, ucds)

    # check primary secondary wording and fix this (and return if fixed)
    newucd = check_position_of_ucd(raw, strucd, valid_ucds)
    if len(newucd) != 0:
        for col in cols:
            table[col].meta['ucd'] = newucd
        return table
    else:
        # search for components in current ucd list
        pucds, pucdd = find_valid_ucds_from_components(strcomponents, valid_ucds)

        # loop around each column affected
        col = cols[0]
        # ask user for correction
        cucd = user_correct_ucd(col, strucd, strerror, pucds, pucdd,
                                valid_ucds['ucd'], valid_ucds['desc'])
        # correct entry
        table[col].meta['ucd'] = strother.format(cucd)

        return table


def deal_with_w50(table, units, warning):
    """
    Deal with W50: Invalid unit string ‘x’
    
    :param table: the astropy table
    :param units: dictionary, dictionary of columns (strings), all columns in
                  table and their respective units (from votable metadata)
    :param warning: the astropy.io.votable.exceptions.W50 warning

    :return: 
    """
    raw = str(warning.message).split("W50: Invalid unit string '")[-1]
    strunit = raw.split("'")[0]

    # find all columns with these units
    cols = find_all_columns_with_unit(strunit, units)
    # attempt to find unit
    attempts = [strunit, strunit.capitalize(), strunit.lower(),
                strunit.upper()]
    unit, found = None, False
    for attempt in attempts:
        try:
            unit = u.Unit(attempt)
            found = True
        except ValueError:
            pass
        if found:
            break
    # if found then set else set to none
    if found:
        for col in cols:
            table[col].unit = str(unit)
        return table
    else:
        cond = True
        user_unit = user_correct_unit(cols, strunit)
        for col in cols:
            table[col].unit = str(user_unit)
        return table


def get_meta_data(table):
    names = dict()
    units = dict()
    ucds = dict()
    for col in table.colnames:
        names[col] = data[col].name
        units[col] = data[col].unit
        if 'ucd' in list(data[col].meta.keys()):
            ucds[col] = data[col].meta['ucd']
        else:
            ucds[col] = None
    return names, units, ucds


# =============================================================================
# Unit functions
# =============================================================================
def find_all_columns_with_unit(string_unit, units):
    columns_with_unit = []
    for col in list(units.keys()):
        unit = units[col]
        if str(unit) == string_unit:
            columns_with_unit.append(col)
    return columns_with_unit


def user_correct_unit(col, string_unit):
    while True:
        uinput1 = input("\n Please type the unit for column={0} "
                        "\n\tcurrent invalid units = {1}"
                        "\n\ttype 'None' to remove units"
                        ":\t".format(col, string_unit))

        if uinput1.upper() == "NONE" or "NONE" in uinput1.upper():
            return u.dimensionless_unscaled

        attempts = [uinput1, uinput1.capitalize(), uinput1.lower(),
                    uinput1.upper()]
        unit, found = None, False
        for attempt in attempts:
            try:
                unit = u.Unit(attempt)
                found = True
            except ValueError:
                pass
            if found:
                break
        if found:
            return unit
        else:
            # print the error and try again
            try:
                u.Unit(uinput1)
            except ValueError as e:
                print(e)
                print('\nPlease try again (Type "None" to skip) \n')


# =============================================================================
# UCD functions
# =============================================================================
def create_ucd_list():
    response = requests.get(UCD_URL)
    lines = response.text.split('\n')
    ucd_table = dict(kind=[], ucd=[], desc=[], keys=[])
    for line in lines:
        letters = collections.Counter(line)
        if letters['|'] == 2:
            row = line.split('|')
            ucd_table["kind"].append(row[0].strip())
            ucd_table["ucd"].append(row[1].strip())
            ucd_table["desc"].append(row[2].strip())
            ucd_table["keys"].append(row[1].strip().split('.'))
    return ucd_table


def find_ucd_in_dict(ucd, ucd_list):
    found = []
    for key in list(ucd_list.keys()):
            if ucd_list[key] == ucd:
                found.append(key)
    return found


def find_valid_ucds_from_components(components, valid_ucds):
    possible_ucds, possible_ucd_descs = [], []
    for component in components:
        for row in range(len(valid_ucds['keys'])):
            if component in valid_ucds['keys'][row]:
                possible = valid_ucds['ucd'][row]
                possible_ucd_desc = valid_ucds['desc'][row]
                if possible not in possible_ucds:
                    possible_ucds.append(possible)
                    possible_ucd_descs.append(possible_ucd_desc)
    return possible_ucds, possible_ucd_descs


def user_correct_ucd(column, ucd, error, pucds, pdescs, aucds, adescs):
        print("\n{0}\n Column {1} \n{0}\n".format("="*50, column))
        print("\n Full UCD currently defined is: '{0}'".format(ucd))
        print("\n\t Problem is that '{0}' is not valid".format(error))
        uinput1 = input("Is any of '{0}' correct? "
                        "[Y]es or [N]o:]\t".format(error))
        cond, res = True, None
        while cond:
            if 'Y' in uinput1.upper():
                res = start_gui(column, ucd, error, pucds, pdescs, None)
            else:
                choice = ask_for_cat(aucds, column)
                res = start_gui(column, ucd, error, aucds, adescs, choice)
            # break out of while loop if res is not None
            if res is not None:
                cond = False
        return res


def check_position_of_ucd(raw, strucd, valid_ucds):

    newvariables = []
    if 'is not valid as a primary word' in raw:
        if ';' in strucd:
            variables = strucd.split(';')
            kinds = []
            for variable in variables:
                for v_it, vucd in enumerate(valid_ucds['ucd']):
                    if variable == vucd:
                        kinds.append(valid_ucds['kind'][v_it])
                        break
            # rotate until S is not at the beginning
            while kinds[0] == 'S':
                kinds = kinds[1:] + [kinds[0]]
                newvariables = variables[1:] + [variables[0]]
        else:
            newvariables = []

    elif 'is not valid as a secondary word' in raw:
        if ';' in strucd:
            variables = strucd.split(';')
            kinds = []
            for variable in variables:
                for v_it, vucd in enumerate(valid_ucds['ucd']):
                    if variable == vucd:
                        kinds.append(valid_ucds['kind'][v_it])
                        break
            # get positions of primary variables
            pos = np.where(np.array(kinds)=='P')[0]
            # set the first one to the primary
            newvariables = [variables[pos[0]]]
            # then add all other variables not equal to Primary type
            for pos in np.where(np.array(kinds)!='P')[0]:
                newvariables.append(variables[pos])
        else:
            newvariables = []
    return ';'.join(newvariables)

# =============================================================================
# GUI functions
# =============================================================================
def start_gui(column, ucd, error, options, descs, choice=None):
    category = find_catagories(options)
    # get categories and their descriptions
    categories = dict()
    descriptions = dict()
    for cat in category:
        # skip if user has chosen category
        if choice is not None:
            if cat != choice:
                continue
        # else continue with all current categories
        categories[cat] = []
        descriptions[cat] = []
        for j in range(len(options)):
            option, desc = options[j], descs[j]
            if cat == option.split('.')[0]:
                categories[cat].append(option)
                descriptions[cat].append(desc)
    # load categories and descriptions into containers for form
    datagroup, flatselect = form_container(categories, descriptions)
    result = fedit(datagroup, title='Select UCD for {0}'.format(column),
                   comment='Select the UCD for column {0} from the '
                           'following pages'.format(column))
    if result is None:
        return result
    else:
        flatresult = np.array(list(itertools.chain.from_iterable(result)))
        flatselect = np.array(flatselect)
        selected = flatselect[flatresult]
        if len(selected) > 1:
            print("\n Error: Please only select 1 category")
            return None
        elif len(selected) == 0:
            print('\n Error: Please select 1 category')
        else:
            return selected[0]


def ask_for_cat(options, column):
    input2 = None
    cond = True
    print('\n Choose category for "{0}": '.format(column))
    category = find_catagories(options)
    while cond:
        string = '\n'
        for c_it, cat in enumerate(category):
            string += '\n\t{0}: {1}'.format(c_it + 1, cat)
        string += '\n\n\t Choose number: '
        uinput2 = input(string)
        try:
            input2 = int(uinput2) - 1
            uinput3 = input('\n\t Correct choice? [Y]es [N]o? \t Category '
                            '= "{0}":\t'.format(category[input2]))
            if 'Y' in uinput3.upper():
                cond = False
            else:
                print('\n Choose category (again):')
        except ValueError:
            print('\n Choice not understood please try again')
            print('\n Choose category (again) for "{0}": '.format(column))
        except IndexError:
            print('\n {0} is not in choice range.'.format(uinput2))
            print('\n Choose category (again) for "{0}": '.format(column))
    return category[input2]


def find_catagories(options):
    # get top level categories
    category = []
    for option in options:
        if option.split('.')[0] not in category:
            category.append(option.split('.')[0])
    return category


def form_container(categories, descriptions=None):
    if descriptions is None:
        lfmt = '{0}'
    else:
        lfmt = '"{0}"    ({1})'
    datagroup = []
    flatselect = []
    for cat in categories:
        Ncat = len(categories[cat])
        # we need to split category into multiple forms
        rows = int(np.ceil(Ncat/FORML))
        for i in range(0, rows):
            datalist = []
            for j in range(i * FORML, (i+1)*FORML):
                if j < Ncat:
                    option = categories[cat][j]
                    if descriptions is not None:
                        desc = descriptions[cat][j]
                    else:
                        desc = ""
                    datalist.append((lfmt.format(option, desc), False))
                    flatselect.append(option)
            datagroup.append((datalist, '{0}_{1}'.format(cat, i+1),
                              'Select from category {0}_{1}'.format(cat, i+1)))
    return datagroup, flatselect


def fix_table(tablelocation, tablename):
    # create current valid ucd list
    valid_ucds = create_ucd_list()
    # ----------------------------------------------------------------------
    # Load data
    print('\n Loading data...')
    with warnings.catch_warnings(record=True) as warninglist:
        warnings.simplefilter("always")
        data = Table.read(tablelocation + tablename, format="votable")
    # ----------------------------------------------------------------------
    # get meta data for all columns
    print('\n Getting meta data (units/ucds)...')
    meta = get_meta_data(data)
    # ----------------------------------------------------------------------
    # deal with warnings
    print('\n Dealing with warnings...')
    for w in warninglist:
        deal_with_astropy_warnings(data, meta, w, valid_ucds)
    # ----------------------------------------------------------------------
    # save file
    if len(warninglist) > 0:
        print('\n Saving to file...')
        data.write(tablelocation + tablename, format='votable', overwrite=True)
    else:
        print("\n VO Table has no warnings :)")


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Run function
    fix_table(DATAPATH, DATAFILE)

# =============================================================================
# End of code
# =============================================================================
