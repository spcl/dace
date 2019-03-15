#!/usr/bin/python3
import gi
import os
import re
import zlib
import base64
import argparse

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, GLib


class ImageStore:
    """ This class implements an image collection for small images, such as
        icons. Storing them in files is not ideal since Glade is not able to
        handle relative paths and different locations of files. Thus, we 
        compress and base64 encode the pixel data, and store it as a Python 
        string. This class can be used via import and the `get_image()`
        method will return a GtkPixbuf, or this file can be called directly and
        it will self-modify in order to add new images. 
    """

    def __init__(self):
        self.images = {
            "edge_tail_redir.png": {
                'width':
                22,
                'height':
                20,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                88,
                'pixels':
                'eJyVVXtIW2cU/1KzJRluK2XqoMRnEanaJCQ32iZoYYrzj5ohOhSCwpw4FJHJKibM4AwVrY/FkPkIg20ybDA+JsJAjFbFdfW5MVqqq/UdXRofdLrEJJq7cwLXhZVSPXC4N+c75/ed79zv9wshhIyPj5OUlJS3NRoNpdfrP2ppacmorKwUSKVSXn5+Pvm/ZWZmEpFIxK2oqIjX6XS3sKa6ujohLS3tXZqmmbQLBoPhw+np6aHNzc3n+/v7jr29Pcf6+vrW5OSkub6+PhFzQ0NDCZ/PJ/heW1srnpiYuLe2tmaF3H+gxmm1Wu2zs7OWtra2W4AZAJhZsL7t9XpptK2tLXp3d9f3fnJyQi8sLPyp1WpFRqORQG+kqqoq9jHY8fGxL2d5eZleXFyksR59Y2PD3tHRoYR9LViPhvGBgQF6bGyMZszlctGDg4Na7BO9v79f5XQ6T9eHhoZok8lEu93uU4wHYIDxM7O3x+Oh4fw0nIlm+j86OsK9NAxub2/vbYfDcYq7urqKZ6KZ3vAJ8xtvampKW1paWmHi/uaB/f549Og3tUYT87VeT2rv3iVfqFRRv8/NzTH9+RtiwFw24DsqYMashoaGGzAPE8TW/rLZXtifPfM8nZnZn9Hpfv1eqZSdECKGr1wAXuwlRLHK55sfGgyWx0+eLG9vb7+w2Wx/r6ysrEOfPc3NzcmIiRciPT2dxMbGvllSUhJR09hIPY2PN1ovXtxxsVi/ANa34D+B3wFXgfeBO+0BAY2ymzc/qKmpoerq6hJKS0sjBQIBJzs7+6V7Sf/n1ccs1thzNvsb6O86/L7ktxYB/sl3ISGKawLBZUokegnnFbgs8Ct2NvuS9OpVoVgiCaSk0tOcKLmcRCQm8qKugQmFAVRCwmtx/U0oFCKn3o+Li4uOiYkhxcXFeHdxXgRiV2DtMuac1yQSCUlNTb0QHh6eGRwcfJ2JBwUFUcC9j+VyOVssFp8bF7jue/J4vFIOh9MYGRnJjY6O5nC53Dvgt3EtOTn53LhlZWUkKysrEO53Xmtr6wPg/v35+fn77e3tDyH2aW5u7jsqlepcmOXl5SQvL++9kZGRH0BXXMg95N3BwYGPh6AzbuCpqbCwMEStVp8JE3UR+drd3f054HgZHk1NTfk4zvD78PCQBq1QY65CoTgz7ujoqJHRDcSFOfi0gMHFGHD0RyhhZWRkvBa3qKiIgFaSzs7O/J2dnSOG+6gH/joG83F3dXV9hj3k5OScaRZKpZIkJSW91dfX9yVqM+iXF3vHHgHbCxprA43TAv8DCwoKzoTJGEVReFfZ+P9iNpu/Gh4evmexWEw9PT0g89obYWFhb8hkslfW/wvK/N6q'
            },
            "edge_head_redir.png": {
                'width':
                22,
                'height':
                20,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                88,
                'pixels':
                'eJyVVWtIW1kQPmoMtKat1epSH1jdqNi8atzSrdXYCMpKC6WPf0pLf4j0BUtRUX9oNBvRClKtKBZbEASR/NiNRIOPqJSmVoMVRY1VY4zvuLFKQeMrOTtzSyTd7db0g49772HmOzPnzMwl5P8RFxdHwsLCvLOyssJLSkp+LS4uvpSZmRkWFBTEEggE3/H8NsLDwwmllJSWlop7e3tfG43GaYvFsr66umqdmZn52NPT81IulwvRJjg42C1Ns9lMmpqaSFlZ2cXJycmpg4MDirBarXRjY4N539/fp+MA1K6vr2diOAoZGRmMXWdn5/O9vT3qRHt7O4U4D793d3dpW1tbKdrm5eUdqZuens7odnR0VDp1MWbIgy4sLFC73e6qq0Db3NzcI3VHR0dJY2Mjc7YGAOb8b+Da2NjYqEwm49fV1bl1DojQ0FDGVqFQXNBqtQ1wlObl5eVPwHXQM3V3d9dBbfDRJjAw0C1NV0RHR5PIyEj/5ORkKdyRCLSESUlJUi6XezoqKuqH9RBCoZAkJiaS2NhYPuA0j8cjWLNAf1g7DyQikejHdcEH6Mfj83m5T5+SlJQUIpVKCcTrAfsIQNcXtd0F/ULvfUICREKh4DaPd+Y3l7iwB8VisT/kE4PfERERbunaPTyI3dPz+DSbrXjj4/PMQYgI11z2JJ/9/Lx/Ons2PSAg4Aau19bWflOLxWLhwyMhIeFExuPHQb1c7jUVm61bJ6QVdG4C/YFsIAvoBXz0gMV6cYzDeQ7zIgQZHx/PQQ0vL68vMUCt5Ofn/6xWq/+A2n03NzdnNBoMC10qlUkrl78yhITcBZ0HwIdIiP/+J19f5fiTJ32ari4DzI5Zk8lkHBkZedva2irLyck5Nz8/T6C2xbCm39nZYWp+ZWXlcA5s2WyO93q9Rnbv3jnQ5AB91RJJ4ohW+2Fve5ux2dzcpGtra9ThcFCbzUaHh4d1BQUFfJgDTa792tLSQqEXDntra2uLKpXK3zEv5J8qVf42+COwp3U6HV1aWmJ0ERgfzJKXfX19bc6ex+fg4CCdmJj4ar5AfkVOXY1GU+46j6ampujs7OyhLsYG/aiEu7wD52FxrrsC9wG/GehncUNDA2lubiYVFRWXYQabXe2d7/iE+1mqrq6+DlfnCbPj2tDQUDf0vxXOdgdoW1xcXO3v7/+rsrLyCsaJdRoTw5QsqaqqujowMKCG/C1oiwTfv/V6fUdNTU0q1gWAyS8tLe0UzIDLsMcdyOFWYWFhnEQi8cnOzv5PXZaXl2PvcYqKin5BW/C5Df+pS6mpqSdhRjM2/wByctyu'
            },
            "unmap.png": {
                'width':
                75,
                'height':
                18,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                300,
                'pixels':
                'eJztV3ko7lkYPubOvWaakpT/UGoME9lHsrsMQ0hChmSpiSn+mDJlQpIkW5IsWbLvhCH7FhMiIVka+5Z9H4z1nfc9zXfHdm++7+Kq8dTb6ft+v+/9znnO+zznPYwxZmZm9o2ampq5qqrqWxUVlZe4ERoaGm+VlJTMxcTEmLKyspm7u/tUTU3NYVNT035jY+NL/BvNzc37qampB4aGhm36+vqMYGlp6VxcXLx2enoKL/gPCwsLEBIS0mJsbKxIPElJSdEgZmJi8mNRUdHGycnJp57iswDxFBQU1KGtrf01csMEkJaW5nyhd7mXlJRs/d/ra3FxkerpD11dXUWsKebv78+uQk5OjobPLCwsvMrKyrY/lq+LiwseN3F2dvbu+8vLy1vv3PVZmDw0np+fizzvpaUlCA0N7TEyMvrW3NyceXl5sbsgLy/P+bKysvqpoqJil+YjKmhvKK5ieXkZOjs7obu7m6/n+PgY0tLSYGJigj+vqqqCurq6a79ZWVmBmZmZD+Y5PDyE9PR0mJqagrm5OWhra+P8CQvKGxYW1mdqaqpCPPn6+t7JkwB4LtLwysbG5ufKyso9Ufmanp6GyclJWF1d5dqnMTc3FwYHByEzMxO6urpgfX0dYmJiIDY2Fvb39yEwMBBKS0uv5aG1j42NXcuTk5MDQ0NDkJGR8S5PcnIy+Qvnva+vT+j50p6Eh4cPoK7UMG7p7n1QV1fnfNnb2/tXV1cfiFLTra2tUF5eDg0NDYBnBtTW1nJeiD/iivZ+d3eXj8HBwfw5rRf351qenp4ezs3VPNHR0byGiKv29nbY2dmB3t5eQO8g/UB/f79QcyX+IyIihlBPmsRTQEDAvXgSQE9Pj0lISHzu4ODwC/ZefwnLV0dHB2AfAtij8FqhtdJaSDd5eXkwOjoKGxsbgD0d5zQxMZFzUVhYCLOzs7zOCFSH+fn5/D3Kg33PtTxUc5Snvr4eDg4OIC4u7hbfH8La2hpERkaO2NnZfWdgYMCysrKE4kkA9DcmIyPz2snJ6Vfcz8O7PPZ9IA2Oj49zPyEd00jrHxkZ4boiL5ufn4eBgQGuIYrh4WGurZaWFv6MQPVCtUS4mYfeuZrn6OgI9vb2uJ7uA/rPqKioUdSPrpaWFisoKBCJJwHI4xQVFV+7uLj8hr57JAxfooI8n7x5a2uL19F91y4MqBbREyYcHR31NTU1GZ4xH8WTANbW1gzzvXFzcwtBLR0/BV8CPMZ/bW5uklb/dHZ2NsI7H0tJSXkQngSwtbVl2JuJ490xDL3j76fk6yFBtRofHz/t6upqqqCgwPA8eVCeBECfZ9jvf+Hp6RmBnnIiSg/zKbG9vQ0JCQmzqI/v0YdZUlLSo/AkAPo8afJLb2/vKOwLTun+SD3+cw/SHZ61Cx4eHj/QOrDXe1SeBECfZ3jGfuXn5xeKntiYnZ1d95wD+7N69PHffXx8rGn+2HM8CU8C6OjoMOTqlaysrLiUlNSzDklJSXHsnd7gvUgM+zKR1vsPWY8H4g=='
            },
            "tasklet.png": {
                'width':
                75,
                'height':
                19,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                300,
                'pixels':
                'eJzdWAtIlmcUfgdjLswYAzfQMZt5ayTYbylFXjI1tbS8X9O8oBKF+80L3tIpTjcFL1liqeWNUjHExJmmljcYzlDE7OI9L4i3SvP669k5LxiDbb+u/NXtgZf/4/OT73zPe85zzvMytjmoqKhgV65cYadPn2ZaWlrM2dmZfnc7ODjsxet9W7FMTEy+wVCkXFxcmEAgYAYGBkwoFLK0tLRN+soPQ3V1NQsKCmIYHzt8+DCzs7Pj3Bw7dmz/qVOn7M6dO/dLaGhozdWrV7uvXbvWJ+l1/fr1vqSkpC5/f/9SW1vbMIzDVFVVda8Uwtvbm3Onq6vLLl68yOLj4yXKDeWNn58fMzIy4nlz4cIFev8ejOkAcuPk7u6eHB4eXp+ZmTlaW1sr6unpgZmZGVhZWdmyJRKJ4M2bN/D8+XOorKxcIP4CAgIq7O3to/X09CzU1NT2ycjI7AoLC2OampoM73Hubt269cG8AAArKSnhfJw4cYJpa2vz+wcPHvwC90YD68zVw8MjDWuuOTs7e+zRo0crfX198O7dO9hpePv2Lbx8+RKwDhYzMjIGg4ODq5ycnOLwu6zV1dVV5eTkpBMTE9mhQ4eYjo4O8/HxYYWFhRvi6ejRo+zIkSPsE4SGhsaXyL2mubm5h5eXV0ZUVNRvOTk5Ew0NDSsDAwMwNze33VT8a8zOzgLlPOb+MtbAMOpEHepFItaLPX7v9/Ly8jL37t3jukL6Ig7GxsaKZ86c8fH19c2KiYn5PT8/f6q5uXn11atXMD8/v92fuumgWqCaePz4sQhrZDQiIqLB1dU1BXlwPnv27NfiuLK2tv4ZtQlGRkZgcXFx02NbWFiAwcFBoLykNTk5+ZdnlpeX+ftJf0iHJBXL34FqheLKzc0V2djY+IrjCjU6dWhoSGKx9Pb2Qnl5OYSEhEBRURHU1dVxbogTAl2TPickJMDr1695LicnJ/P417ibmJiA0dFRicVIaG1tBZxDfhDH1fnz51No3yWFpaUlzgdqHzx79ox/c01NDfUtePLkCTQ1NUF/fz+g3kJjYyM8ffoUcCbgsWMPgYKCAkhPT+e/xJ2k0NLSQlz5bSdXhNXV1fdcjY+Pk85CXFwclJaW8l/q+ZcvXwacIXntYf+Cmzdvwp07d6CqqopfE7+SxE7hijQL51hoa2vjMaWmpkJ0dDT1dbh79y7gvAKxsbGct4cPH0JKSgrpB+AMyZ/Hns6fo/yUFDbCFenV8PCwxGIgEFcdHR0wNjbG51W6phzr7u4G0sqpqSl48eIFUBx0nxZpVHt7O+/309PT/O9rGicJbESvqA+WlZWtUG79H2eE9bA2f+Esv4xeSWwfPHnypIKZmZmrp6dnWmRkZBP+z1h9fb2I9HYnzuUfC8prymes9eUbN24MY3+uQ++diHOoHXqjr8Rxpa+vzz0N8sPQe+7B6wOmpqZObm5uSX/2e9T7aQ/+S6CesuYXHzx4sIT9dBB1sxprLQ7ndmv0cCrKysrSly5d4nM7zlfiqHoPnIG4r6RzDfLKqL3E3W70Tfsx9+gcIQH3oBb3Yoj2hDwXea+dBOKGZrSuri7A+XoB+2o/eulfHR0dY9APWiA3SoqKirtQo7mXpnOBwMBAhj5uQxz9E4qLi5m/vz8zNDTkPpN4VFFRkRYIBCr4Divcm59wj6por3DPFmnvaA+pV23VotmN+kJnZyfNu/PYP3uEQuF9rKXI48ePmyE33ykoKHxuaWnJv4H8Hp054EzyUdysh9u3bzP07KRv/L2YZ0xJSWkXcrcP+TR3cHD4EX3pfXyuMS8vb0sW6kMDzmWFyE0Ien5j5OZbWVnZz9ZipHNInOsYzrES5WY9ZGVlMfSdDPsCr3UrKys6H5JCHy5jYWGxJQu5kMFQPqWzFdIN9L6cG+zrm/KNfwDeM4BN'
            },
            "stream_unmap.png": {
                'width':
                75,
                'height':
                22,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                300,
                'pixels':
                'eJztmOsvc0EQxv934kMTH0QibnEpglQiEiH4QgQtJa2WuEYogrgVperSkd8ks6mqS+qceL31JCdtn+2emZ3dnXl2Rf7wHTw9Pcnj46N+5vP5kryB9vd4uI/45+dnx/P9M77Ql/d4P303PDw8yOXlpbYNDw9LY2OjtLe3y8nJibbf3d1JKBRSvru7W66urpRPp9PS1dWl/MDAgNzf3yt/dHQkbW1tyo+MjDib29vb0tzcrPzU1JSzv7y8rBzP4uKi42dnZ5VramqStbU1N5aJiQnlW1tbJZVKuTEMDQ0p39HRIWdnZ8rf3t5KX1+f8r29vZLJZJQ/Pz+Xzs5O5QcHByWXyyl/cHCg74UfGxt7NXeA/i0tLbKzsyPX19dqh3cRW5tHYgl/cXHh+hMDfsMTN5tL+tEfnvcZjz9wPOYzyGazjmdeDIzTeJsHUOgjMbIYMofmo81Poe98fuY77zMeO6WwsbEh9fX12q/SQQxYu7ZWSoF1VemxYi2zB6enp1/lvD+8BftrfHz8y3EiX87MzPjs1b+J4lz+Gci71IFIJFIx6/D4+FhrTDk4PT1VHVBu/6+C/Ento84V26JWUYv4D3nEr3kjTzc0NDgtVA6KNZ0fYE7QXEtLS9LT0/OqjTjV1taqLkOPfVSXygU2iBN5ygsU6x4vgZapqamReDwudXV1b9rRj+jazc1NX+wzLvSnV5ifn1d9brrPS9zc3EhVVZX09/dLIBBQW+vr6659b29Pqqur39WF34Efe4a9yBkHfeY18Jc1k0gkNG8sLCxINBp17dQl9qnX45qcnPRs3xWD3Msc+52/sFPqzOoV8J/6Tp1nTfuN36wjyCOjo6O6Vv0Ge4K96GU+/J/BHQrnbGrYbwH6en9//0dsz83Nacx+A7gLCgaDP3puI2+R78PhsOZLuxsEW1tbyqMBuDcC7N/V1VXludOzOyxyCHGHj8Vi7q4K7YNGhU8mky7no6+pl/DoC8uf6ED8gKemGtCx1HE/a8ZXcHh4qOPjsbtIsLu7qxwa084OxIoYwjN2zi0APc69JzyxtDtKzjwrKyvKU99trNQvYgtPTCxW5ATswRfuN/K4vfMP/uMFOZ4pMQ=='
            },
            "stream.png": {
                'width':
                75,
                'height':
                21,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                300,
                'pixels':
                'eJztmNdKJFEQht9XvPMBvFBEQVHMigkDJhQVE+YsJsw556yYs7V8BdWMu7rBnTMzG35ohq7p013xrzpHJPB4eXmRu7s7ubq6ktvbW09+fX0t+/v7et3c3LyRX1xcqIy1fwOenp7ULrMHX3R0dEhVVZW0t7fL8/OzypeXlyU9PV1SUlKku7vbWz8yMqLyzMxMmZ6eVtnr66u0tLTos8i3trZU/vDwIPX19VJWViatra2eb9Hh7OxMY8DaYAMd0OXx8VHv0bu2tlbtKSgokMvLS5Xz29fXJ8PDw7KxseHpzjr8iH32DnsvfubytRP7yUPiYP7mme3tbZmYmJDR0VEvP/FTfn6+JCcnq06sBXzr9PTUuw8EpqamJDc3V9LS0mR9fV1l6L+zsyPHx8dyf38f9HjiD+KEb0yXlZUV1ZlrfHzc79/E7pmZGZmcnPRk+GdpaUnzItg++VWQwwcHBxpTgP5dXV3S1tYmR0dHn7bn5OREsrKyJC8vTxYWFvypckhhd3dXeY9aweafBbVk/EFOsdbfuWOcFGrAbtOLvPue36jx6upqGRsbc6YP36DvNTc3a94Tm1D0GxyTmJioHPce0L+4uFh7mivACREREbK2tiaDg4Pa6+mTyOFBZqi9vT29p+/RL5ARY3oe8wKyw8NDZzoamGuoS/TwBbFFd/RyCfIoNTVVwsLCZG5uTvmwvLxc4uLidNaoqanR+9jYWJ2p6urqpLCwUDIyMqSoqEgiIyP1maSkpDezhivgj0B85z3Y/N3Q0CDh4eGSnZ0tPT096j/ynjmUGp2fn5f+/n59Lj4+XuPLPXXBjIov6cOusbi4KAMDA9/IyS343CWos5KSEp0dyQ1yiLzBdmYT8igmJkb9l5CQoL6MiorSZzs7O9VvQ0NDEh0dLefn5051hVtzcnLencWoS/Kc2nAF8hnegQvIL/KMeR4esnxnPoGfuDY3N/U/LuZKuIq5nHe45FWAH9ib2T7haxB39gZw178Icsn4id8f7YeItfUZ6jLQe6hggL0itU3Nw5WfAb6trKzUPRR8Eaye4G9QV+SEzdlwJdy5urr6Yc397HvhB85LjPvZU3HWArfYmUKoAn+YTziLoLfSbysqKrzYu9zX0oOYIzn78D1/ou8zW5LHvj3dpS68G46gjozvkbH/KC0t1bmN2AL6F+ck7AFd94aPdDWQw/R95n/8BuC8xsZG7b/EEv4D6Io99JHZ2VkvxtjDXENdMN/b+9mjEZempia110C+M59xdsYa04k5jH5P7/yT+JY+T4+lZu38El8xX2I7uWj1TU339vbqOSq22n6R3ECGv/ChgfzF/7z3d7jmP/yLL/AYyXE='
            },
            "stream_map.png": {
                'width':
                75,
                'height':
                22,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                300,
                'pixels':
                'eJztmNtLskEQxv/1IOgqgurCiM4gdFOBUhiYEGoQnQ+EEVSUWdlBO5+bj9/ALK9SWR/vakIPLOijuzs7MzszOyJ/qCfe3t7k/f1dPz8/P8vl5aVcXFzI9fW1+8/j46NyjLu7O8ff3987/uHhwfG3t7eOf3p6Uo49rq6ulGOPl5cXt3+5XFa+VCrpd/D6+upkYZ7J2ChwpvHxcbm5udHvR0dH0tfXJ5FIRBKJhJN7e3tbenp6lM9kMm7+wsKCcoz19XXHp1Ip5Xp7e2V3d1c5dBOLxZQfGBiQYrGoPPoeHR1VPhqNOlnQ0dDQkPITExNqx0YBe4+MjFToBNtxJgZ2NfC78fbfr3jmGh/0h+/whmpZ8PP9/X0vuqiFk5MTmZ6ertDJbwb3tLu7W3283mj0/f8f7O3taSywe+obx8fHGkObFefn5y5f+AR66uzslEKh4H0v36iOeWECWwwODsrKyoqX9esN8i55yYe+sMPp6Wmoa5L78FVyFL5anSMPDw81tnBvqNHCBHUeNQk1TJj68pXr0EdXV5csLy9Le3t7Rf3Kb5xlZmZGZmdntZ4MG9SqrB3W+bLZrMzNzYWy1kfo6OiQ1dVVaWtr05wRtPHW1pa0tLSoLn87Njc3NZZTA/sC9Wx/f7+0trbqfaA2N31x78nxBwcH3vY35HK5Cr/+CZAXnyJm+ARxI5/Pqz2IXfPz8xW+RZwKxjFfSKfT+nYKOy76Ajqph14+25u3bfC9WgvoFX9slMyNBHf+u/eQfDA2NibxeLwp3zBhgbPXuosbGxsyPDxclzfAbwZ6or/zVaymH9Ussc03qAGo/T56+1KTU/NPTk5qvWOg5wPHCPYzqH2MN/3ju0tLS8qx1tnZmfLEAXIqfDKZdO98+nT09eCpCc2fkY9+Dzx50WIneXJqakr5YG+QfqDJsrOzU3FeOOaQawFrLS4uOhmtHqIXSFyHp/4lP6+trWn/shr0cOEZ1LMGfM34YNyjH2R8sC/M3nC8WcxHkY8+JjzntZ4c8ZE+GDy2Mp2gM+bD876x2Ilubc+gvdF9LRmZa/YMymj2+UzGP/wc/wCzIy7C'
            },
            "state_trans.png": {
                'width':
                75,
                'height':
                16,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                300,
                'pixels':
                'eJztl1tIVFEUhndZ5CBRllBkN0shspLQyMzS0YjKysyZpixSsqvpJKZRViRJWFkmDFEahIl2sRepjBjIy4P45pMipZBXfBCFyUJQ4+/fJ4dIxlPRrvGhDYszcy7rnP87/9prHyGESEnBskOHYDx48M9jLE/k/v3wFW4Ye/difkICIlXqSUrCMmf+q1dx49UrON6+VRPPn8Nx8iSuTKRn61ZITb+sv6QE4uJFCKsVIi8P4to1iKoq19fHxODY06fqtLx+DcelS7jhzP/gAWyfPkHZ6O0FLlzAbVdaPnyAZDU7Ph6GwECI7GyI9HSI5GSIEycgsrK+MZBsYmMhUlMhHj2CB89bcvkyZly/Du/cXCymhimu8tPPaV1d6rQMDQFFRbA58xcXw/bxo7r8PT0aq1uutHR0aKwCySooMhKBBQUwkNVsen0GPe9NP05buxaCacTu3fDNzITPvXuYFh6O7KAgnDl6FMHkmlRWhqmu8lssSOU9lA3pofv33cOKx6bu2AGL0Qjrhg3IYU3l0S+V3JcbHY10zg0+69ZhKY8fiYhAIhku4LOKVatgZNzZsgWZ588jhnWu5UtMhMjI+F6Pk4XVyMgIOvgg/f39uufpsXr3DiIsDIfXr0cGtZ/j73z6yrJnD24uX45QWVvBwYimj/LJbjN9ZjCbIVavRsSmTVgREoKyAweQ/D0jtJD1TE6/xaqvrw8Oh0Mpq+HhYQwMDKCurg7V1dWoqKhgjoknOMmK88tN1pgYH1LXmjUI27gRUdu2wUJvJZCThVxiyW8l9wtfX4ioKMyh17bTM37yGvIz7toFH57rf+oUQmQuek94eyOA52fRi6H0pBd74Gk9VqOjo5qWIU5EVWwQDQ0NSlk1NzdrfGw2G5qamlDMC7q7u3VZHT+Ol+xJZp0wUbuZYdq5E/vIxTT+OPfHM7T/PK5t+V+7buy32d8fuR4e+OLpCcesWaim9+ydnRNr6eHDlZaWatuamhrU19crZdXJmxexGRQWFmq5y8vLdb0rWbF/PWM9GP9msOcZAwKQOX06Pnt54f28eXhIfz7R85WcP0q4CKmtrYXdbtdqRSUrmb+trY1rgV60tLSg6ydNeawG8/nexd8M+kv4+WERvRXHWl7IW09hDabosRocHERra6s2V0lN7e3tSln97tCb29UPiJkzIUJDv61XJ0sf/NXxb1n9OP6zmjysOG3fVclKtkiuFwvcxMqqmhW/G+468/Pb+WxlJRrfvFET/HZtTEuD1R2s+E1uefxYnZYXL9CYk4NzzvxcNxtMJsyNi1MTfLdzWYMGd7Bi//WU91elRXKhFi+Z+ytNqw0/'
            },
            "state.png": {
                'width':
                75,
                'height':
                14,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                300,
                'pixels':
                'eJztlTFqAkEYRjc5QdpcwKClRcADeIW01mIlFhFsXQnBxi6ygoWKoIhiIYIgaCmI3kAUxCjbuIogii87S4QsMZBup5gH38wwTPHz+GdG0zQtHOYhGiWWTJJUcSeR4DUS4Vn7JhDAV6uxms1AxZ3xGFIp9Ksrvx/fcMgnil+YJmQyvP10NRiw8rouGdlsQNflcbXdbpnP5xwOB69K+BOZXF0uF+r1Ot1ul+l0ymQyYb1es1gsGI1G9h0wWS6XzjkvkM2VcJTL5Wi1WjQaDXq9Hvl8HsMwqNmfjvB4PB69KE8qV+fz2f5rxnQ6HdrtttNj5XKZQqFApVKhWCw6vXY6nbwoTypXgv1+77xZwsdut3PWYhb7lmU5s1fI5kpmlKv/c8PVU7+PKZ4EFXdWdgel07xfXYVCPGazfDSbVFXcKZWoxuO8XF3ZrSbGe5XbCQa5E56+AI/z9Ck='
            },
            "run.png": {
                'width':
                27,
                'height':
                30,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                108,
                'pixels':
                'eJy91s0rBVEYgPHXZ/koKeWyU5JslLKRkpW9xU3JQv4AKUlZKMoSC2UlOxaykJW1zY3ETlIohCvKwlc+n9PcU6N7pzEz55y3ftt5Fu/MmSMitSKygGscYw5NYmcm8IafnC8cYQQ1hlv7vo7fO3bQh3JDrduAlvaEFbSjKGErG9LSLjGNRgctvcsDDKHacktT79I2elFmuaU9Yhlt8r9dJmlp55hEykFL73IPA6i03NJesIlulFpuafdYRIuDlnaKYfHOHtstRZ09Y45aypXDVtZR6xWzDlo3GEeVxdYz1tCJYvHGdOsDu+hHhfwdU61vnGAUdVJ4TLTUM+bRHNAw0VI72UAXSkI6cVufyCAtwWe6idaZePe8+giNqK0HLKFV4t+n7kIa6pvfQo/k/4+izmFAQ/1n1T11UOLdmQrNjHi79ncuMIUGQw096nnruZ7aySo6JPkdN29+Abgq6aY='
            },
            "reduce.png": {
                'width':
                60,
                'height':
                22,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                240,
                'pixels':
                'eJzlV1lMlFcU/sGWUBt9sI3pCzRN0ybFN5OmLT5QMGmAAYbNASMwJSADSNkKOGVfiwpmBBUIFULCEtkUFB5sgWFX2VelwzKDCAyEgYGI7JyecxOJbSJNGAYbe5Kbublzl/Od853lj4+P/1IkEhl7eHhoZbi7u39nZGT0hZ6enqG+vv6nOjo6hgYGBp8LhcJvtPXm68Pb2/vUr0lJnx0+epSbev6cMzExyfPx8VFHRUXNR0ZGamOoLl68KA8PD5fhfBjnMrFYPBYXFzenpfd2BmFCvEs8Hu8ahzI5OUl4yzo6OmB7exs2Nja0NjY3N2Fra+tvg9a0+SZhevToEVhYWOQQ3omJCc7U1LSsp6cH3lUhX/4Tb29v79tWS2vS2dkJyOcdvGZmZuX/J7zW1tZ/aIJ3dXUVnjx5An19fTA3N7fr3pWVFbaXYveghPBaWloyvGNjY5ytra1UE7xDQ0OANQfS09MhODgY0IZsfWFhgQ3KGSSLi4sgk8kgPj4eXr58Cevr62ydfl/lL5VKtXOG1mdnZ5mNNMVrbm7O8KKtOTs7O2l/f/+e7yO/Yq2BpaUlCAkJgYcPH7Jx+fJluHTpErS1tbE3sf4wrAEBAUD2LSsrY+cLCwtBLpdDRUUFJCYmQk5ODkxPT8OtW7cgNTWV2ZFssFfp6urayVeE197eXiO8AwMDwOfzITQ0FLC+w9TUFPj7+zN9MzMzITo6GhISEqC9vZ3ZhmzS3NwMaWlp7DzZRSqVQlhYGMzMzLA60tLSAp6enlBdXc3s09DQsGf9uru7d/hMeB0dHZs09S/pSjyOiYmB2tpa8PPzg6KiIqivr9/x7eDgIIyMjAD2GtDa2sp8R9iwL4Camhq2ThwhIXzYJ7DzdXV1oFQq9wfvwADn4uLSqwleil+JRMLmxN3r168zbiYnJzNuEjbCExsby/iNPQ9gn8Ni4MaNG+Dr6wvDw8OQm5vLfE12Gh8fh5SUFMaPvLw8jfEih2/jVHdcJuNcXV37iJN7FcorlItIqF+iWCO/kc5kC/IZ5SKFQsE48Gov5aLR0VGYn59n+9fW1hhu2kf7X7x4wc7TPfT/XoV6KScnpzs4PTQul3NOAkFHdnY2e4vy5rsiy8vLrB4QRzAnF+PSoWejo9wFb+/vsQYnoA1qAwMDlTdv3tykuCHOaWLXgxbiGelMuQ/jZBPz3DT6soZvY5PoLRJ9TfH7bGiIUykUHG7nJFevHnZzczPiW1u72tjYZOG8KzIiYrGgoGCbcvrrtfS/IKQLxQLlw/z8fIgID1eTzqh7JmI896NQ+BXmlQ9wK6dWqbg3SUtzMycfG9MVh4UdQ58bYy8W6GBvX+7l5TWC+Wb1/v37b4379CZx9N69e5QPV0gnzEWlVlZWAajrt4j5mHJiQrfz8eM34ttNyDY/nD7NlRQXv4+cMLSzteXh3UnOzs7SoKAgZUZGxlZjYyOrudrgPt1Jd1N9wjjbonhDXHXWqAPqYvGTn59heUnJexfOn2e67rfMK5Xs3jSJ5ENXF5cT+K4b9hq/Cd3curHOLFGvRPlQrVbvift0huKGagjGEdXmReIovpFtg3HmLhSeuIZvkw4Lc3P7ju/fBHnF/fn0qe4vYvHHZxwdT1nxeD87ODjcEYlEo1euXFmrqqpiPcZuPTBxlOKD4gTr7yqdpfjBHBosOHPGGO/+aFyh0P39wYMDx7ebkM0/OX6cKy8t1fPy9DTE7xAr9H8ycr8OvyOUWVlZG01NTaw3Jo5SHGCt2MC4mHbG2oB5JtmWz+dR3FTevasnsLPTCke1Jfitw/RNTUk5cu7s2ZPIR3/kZaVAIFDjWEB7VKIP/bDfOSlJTT1CezEOtK7XX65D1f8='
            },
            "map.png": {
                'width':
                75,
                'height':
                18,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                300,
                'pixels':
                'eJztl0lIY1kUhp9Y9MK02AvRNK1gGjcRxIUDuBAX3QgGcQZHnDViKs4TunICjbTiRmjcutBOnCecR9oRFUccUMFyJQ6JRu04nD7nwpNY2FVJWWUp9g+Xl7x33r33fPfcc+7juKfJ3d2dc3BwEDg7O//m6uoqeanNzc3NOzg4WOTt7c2dn58/0WvjReOKRCKz2NjY8ubm5tOxsTHtS239/f3aoqKisbi4OLFEIuFyc3OfjVNoaCjFlBmOrZiYmNDBK9Du7i4UFxf/LZVKxTY2Ns/CKTExkfPz8xPgVYFr9io48drZ2YHCwsI+W1vbnx0dHb8pp6ysLC4+Pl6QnJxcOTIyoru7u/ve7hut7e3tu/T09L8sLS2tMZdx34JZfn4+l52dLZDJZJXDw8Of5XR2dgY63cOwu76+hpOTE7i9vWX/Ly4umB2J+qNn+u/QPbVaDR+PpdVqQaPRsN83NzdwenrKbC4vL+9/f0pbW1tAvOzs7Kw9PT254+Pjr8bJ3Nycw9g1S01NrRwcHDQonjY3N+/94f3u6+sD5Azj4+PMx+XlZcA1YL5TPsG1gMPDwwds19bW7tmSjo6OoKurC7q7u9k7xKeiogJ6e3thb2+PjWHI/DY2NhgvJycnay8vLw5vPZmTUCiki2l4eHgu1pN/9Of9XyIbmvPKygpg7ofR0VGYnJyEqqoq2N/fB4VCAVNTU8zXzMxMGBgYANzTkJGRweKIF9Z3wBoLMzMz930QE7pHvOvq6mB1dRU6OzuhoKCArcPS0tJn58drfX2dxldibAn9/f2fxEssFjNOPj4+sp6eHrUhnHjV19ezuatUKmhoaGBxUF1dzVjRlfbewcEB8zMnJwdwHaCkpITtQz4urq6uoKamBjo6OlgfLS0tLKboSqyampqYzfT0NItRjHsYGhoyeI4kilvMw0r0UYjx8EW8PDw8GCdfX18Zzs8oTuRrY2Mji4OFhQWYm5ujnMp4UHyQn4uLizA/Pw9YS5nPFA/l5eXMnveXcldtbS2zoz7oHWJN3KkvsqVn7e3tLAapL+rfWFH843qpwsLChAkJCUZxCggI4CwsLEwDAwPf45qqKbcYI2JFc38st9MeI+6Uo6iRHdnTM8rLmGcZV7IhH5RKJXumL6oJ9B7Ni66UF/k5fmxrqGit8IyqSkpKEkZHRxvECc8EnIuLi2lISMj7trY2jbGcvpaIH+1R4vdcorjNy8trwjORtZWV1Sc5YRxyQUFB7yIjI+Wtra2aL12j1yza0ykpKX8ijh/xG+5RTmVlZRzWoXcxMTFyrDNvkhMvzI26qKioP0xMTASPnVVLS0t/QJ7pWGPOv9e+e0lCXjeY56vt7e3Nsc4+YBUREfF7WlraOp6LPuBZ5wPW+zfdiAHutV2JRJKKddtUnxV+C/+EHH/FXCXC2vnmG561RMhDhGfVX/AsbKLPSi6Xc1Kp9P+m1/D8wL6DZ2dnuX8BB0hemw=='
            },
            "edge_thin.png": {
                'width':
                75,
                'height':
                31,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                300,
                'pixels':
                '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'
            },
            "delete.png": {
                'width':
                23,
                'height':
                30,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                92,
                'pixels':
                'eJxjYMAJPID4ORD/x4OfQdXhAlFoOBKIM4H4GgFzYfgaVH0kFrP+4cDEmAvD1DCDVHyfhlgBiOWAWBqIZaiE5aDmgoAilnigBCsyIEA8EP+lIo5DM5uacRg/avao2aNmj5o9avaQNzuBhmbHUdls5DrNDYi/UslckDmuSGYLAPEmKpm9EYj5GVCBBhBvAOJvZJoJ0rceag4YAAB83CNR'
            },
            "cursor.png": {
                'width':
                30,
                'height':
                30,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                120,
                'pixels':
                'eJzNlj9sElEcx4//JhYcSGkgBCYn4mBkMOzK0I0Bki7deg4kBOaqzMbAhHFx0DA4OZCwkEA7wtIuQNyAYBzocL0/7/jX+vz+bGM0ViVyvPhLPpf8Xu7uc+/d797vJOm/jHtgH2wJ9h6CS/AC3BboLdpsNg7mgt3FQCDAU6kUc7lcM4HuYjQa5b1eb5nP5w2HwyHK/TwSifDRaMRN01zmcjlR7m/e4XDIKQzDEOX+ySvQ/YtXkPtGrwD3b70bdv/Ru0H3X70bcq/kpWCMXRQKBeZ0Oq1wr+ydz+fLRqNxHgwGdVy3kK762Ka8XyaTyaxarWrJZNLw+XxLXPMZvJOueqjV3ks6tNttPRaL6XivDOeegmfXPtcazu9e2p9pbqqqzmq1moYaUvr9vtlsNlWPx3OB896C7TVdP0YxHA7zVqvFyuWyGo/HTbfbTb14IcuyhueYJhIJE/kxuGOlF32X+/3+BXq/hvwIPAHvQ6HQdDAYGJVKRUVOtbRroVcGA/AGPAbe6/FHQC2VSup4PGbo0bQGtNZOi7z0Pxe54X7kb2LdmaIoZjabpTl/ktar4VVDxrue1et1vdPp6F6vl77ZpwK8UfAxnU7r2CdN+n6Rn4AdAe6XVHPdbtegvcNut08xtifA+xCcYc5KJpPRsH9w5K8FeG+BD4B8VPevwH0BXooH4ADcBQ5Bzn+Or6ajC1U='
            },
            "array.png": {
                'width':
                75,
                'height':
                21,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                300,
                'pixels':
                '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'
            },
            "conflictres.png": {
                'width':
                75,
                'height':
                28,
                'has_alpha':
                True,
                'bits_per_sample':
                8,
                'rowstride':
                300,
                'pixels':
                'eJztWGdIZGcUfeOO+ydodN0lm92QYPmjf/wTggS7YBd7LyCKYkWwoosRLEQ0QWJvKDZGFMVVEY0aA2os0Qg2sCCixhLsFXXmy/keDsgGo1P0mY0HHg/e6Lx7z3fuufcOwzzhCU94TMjJyWEIIQp6enr6ysrKTmpqag5KSkr/m0tFRcVBVVXVEblbq6urf6KhoXEjVw0NDSxXVlZWsTo6OjtxcXGkurpaiOuyqqrqo75qa2svoRWRgYHBpa6u7nsfHx81Jyenf9VWRUUF093dzXd1dbXy8PAYqqurEx0fH5OPGSKRiIyOjpKoqKgt5P1dVlbWS+iM6ubWWlxeXma0tbUZPz+/N+bm5lkxMTG7k5OTXKd0L9je3iaFhYWXzs7OPdCS0enp6bO+vj5J7YtJSUlhOjs7+Q4ODjZubm4jNTU1osPDQ67TkwuEQiEZHh4m4eHhmzY2Nu+Sk5NfgS+JOboOcqXD0NDQt9bW1j/Cw/ampqZY3coKeXyHNKBays/Pv7C3t+9CzRlsbGwoZGZmysTTdZSUlDCNjY2Kjo6OdtDY7/B7IovGaLz0XAcGBsj4+DhpbW0liJkIBAKytrZGmpqaSE9PDykrK2P/hr5vcHBQJo6oloaGhkhwcPCKkZFRXFhY2AstLa07+ZKkWFpaYtBLmZCQkC+g2xxobJ9qTBpQHuChZGZmho0/IiKC7OzsEHw3+wy1QTo6Okhubi5JT08n/f39JDo6mpyfn0v1Pno2BQUF5+jxrYaGhl/jEc/X11fuHH2I1NRUpre3V9Hd3d0evXIM/VZ0dHQkUeybm5skISGBvahu0IPI/v4+y9n8/DxJSkoiExMTrL6QI1ldXWU/k7Qni30JZ7Cqr68fjZg/5fF4DJ7dO09inJ2dsXec9Zeo+5/i4+MPpqen75wD1ePs7CxJS0sj4JoEBQWxNebt7U3GxsZYDdEc6WfZ2dlkYWGBBAYGkoODgzu/40pLF/DZdtTcN1tbW7zy8vIH4+hDYP5iUCvPoS8nT0/PP9Ar7+Rji4uLLDfUq/b29li/old7ezuZm5sjzc3NhM4pVHNdXV2s1urr69k6vQ3XtPQntBQL/1a5ba58KKysrDB0doMWvoLG8mNjY6X2MVlBucS8dGFnZ9eB2VAPtc7D3MQ1Rf8A9THM/M/h+9aYVX6TxsekBdXSyMgIQW9bNzExiff391elMaF2uablRuBc2buXl9fnZmZmmdDYtiQ+Jg2uZm+qpU4LC4tvd3d3FTDfcMzE3YF5gkH8fJyxBWayAfiYUN4aE/sS5uQNvCcRWnoBb+I6damAemTvqMnX2N8zsFfKTWPUl4qKii6xf/0MPemDMx60zHHGsgOez2RkZPCNjY0t0I8GZfExsS/RPQ7e/Q53NfoO1B3XacoN6+vr7MwPrl6jXr5Hje7Q+VxSLRUXF1Mt9aDfGtI9rqWlhevU7g3wFgb58k1NTa2ueuWtPnatx23Bu5OhpZf/VV+SFCcnJ+wd+b6BxrLQK3dv8jGxlsDrLy4uLkaYTRUwv3OcwcMD+w2Tl5fHt7S0tAFvw/Q3WLHGqJbob5WRkZF/2drapmCHekX/RyQScR02Z8DuwmhqajKYrd/Cq38AJ3u03kpLS4XQ0a/YdU2xNz9ra2vjOtRHA3gQU1lZqQiN2WHPrYd/pyQmJn4GrrgO7VECpcdgj2YCAgIUBAIBD7sv1yE9QQr8Dds3KLI='
            },
        }

    def get_image(self, name):
        # get the data back: decode b64 and uncompress the pixels
        data = self.images[name]["pixels"]
        data = base64.b64decode(data)
        data = zlib.decompress(data)
        pb = GdkPixbuf.Pixbuf.new_from_bytes(
            GLib.Bytes(data), GdkPixbuf.Colorspace.RGB,
            self.images[name]["has_alpha"],
            self.images[name]["bits_per_sample"], self.images[name]["width"],
            self.images[name]["height"], self.images[name]["rowstride"])
        return pb

    def add_image(self, filename, name):
        image = Gtk.Image()
        image.set_from_file(filename)
        pixbuf = image.get_pixbuf()
        if pixbuf == None:
            print("Couldn't load image " + args.file)
            os.exit()
        colorspace = pixbuf.get_colorspace()
        width = pixbuf.get_width()
        height = pixbuf.get_height()
        rowstride = pixbuf.get_rowstride()
        has_alpha = pixbuf.get_has_alpha()
        bits_per_sample = pixbuf.get_bits_per_sample()
        pixels = pixbuf.get_pixels()
        pixels = zlib.compress(pixels)
        pixels = base64.b64encode(pixels).decode('utf-8')
        newdata = {
            "width": width,
            "height": height,
            #                ignore colorspace for now
            #                "colorspace"      : colorspace,
            "has_alpha": has_alpha,
            "bits_per_sample": bits_per_sample,
            "rowstride": rowstride,
            "pixels": pixels
        }
        old_contents = ""
        with open(__file__, 'r') as f:
            old_contents = f.read()
        p = re.compile('(self.images = {)')
        new_contents = p.sub("self.images = {\n" + "\"" + name + "\" : " + \
                str(newdata) + ",", old_contents, count=1)
        with open(__file__, 'w') as f:
            f.write(new_contents)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DIODE ImageStore')
    parser.add_argument(
        "file", metavar='file', type=argparse.FileType('r'), nargs='?')
    args = parser.parse_args()
    ims = ImageStore()
    if args.file:
        ims.add_image(args.file.name, os.path.basename(args.file.name))
    else:
        image = ims.get_image("conflictres.png")
