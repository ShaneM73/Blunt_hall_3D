import struct, sys
def varint(b,i):
    r=s=0
    while True:
        c=b[i]; i+=1; r|=(c&0x7f)<<s; s+=7
        if c<0x80: return r,i
def dec(b, ind=0, path=''):
    i=0
    while i<len(b):
        k,i=varint(b,i); f,w=k>>3,k&7
        if w==0: v,i=varint(b,i); print(' '*ind+f'{path}{f} int {v}')
        elif w==1: v=struct.unpack('<d',b[i:i+8])[0]; i+=8; 
        elif w==5: v=struct.unpack('<f',b[i:i+4])[0]; i+=4
        elif w==2:
            n,i=varint(b,i); s=b[i:i+n]; i+=n
            try:
                if all(32<=c<127 for c in s) and n>0: print(' '*ind+f'{path}{f} str {s.decode()}'); continue
                print(' '*ind+f'{path}{f} msg[{n}]'); dec(s,ind+2,path+f'{f}.')
            except Exception: print(' '*ind+f'{path}{f} bytes {s[:16].hex()}')
            continue
        else: raise ValueError
        if w in (1,5) and v!=0: print(' '*ind+f'{path}{f} {"d" if w==1 else "f"} {v:.10g}')
dec(open(sys.argv[1],'rb').read())
